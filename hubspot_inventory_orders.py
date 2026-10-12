"""
HubSpot Inventory & Orders Manager
Uses HubSpot's built-in Products and Orders objects
"""

import os
import asyncio
import httpx
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from loguru import logger
from dotenv import load_dotenv

load_dotenv(override=True)


class HubSpotInventoryOrdersManager:
    """
    Manages inventory (Products) and orders in HubSpot
    """
    
    def __init__(self):
        self.api_key = os.getenv("HUBSPOT_API_KEY", "")
        self.base_url = "https://api.hubapi.com"
        self.timeout = 30.0
        
        if not self.api_key:
            logger.warning("⚠️  HubSpot API key not configured")
            self.enabled = False
        else:
            logger.info("✓ HubSpot Inventory & Orders Manager initialized")
            self.enabled = True
    
    async def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request to HubSpot API"""
        if not self.enabled:
            logger.debug(f"📊 HubSpot disabled - would {method} {endpoint}")
            return None
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if method == "GET":
                    response = await client.get(url, headers=headers)
                elif method == "POST":
                    response = await client.post(url, headers=headers, json=data)
                elif method == "PATCH":
                    response = await client.patch(url, headers=headers, json=data)
                elif method == "PUT":
                    response = await client.put(url, headers=headers, json=data)
                else:
                    logger.error(f"Unsupported method: {method}")
                    return None
                
                if response.status_code in [200, 201, 204]:
                    logger.debug(f"✓ HubSpot {method} {endpoint} successful")
                    return response.json() if response.text else {}
                else:
                    logger.error(f"✗ HubSpot API error: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    return None
                    
        except httpx.TimeoutException:
            logger.error(f"✗ HubSpot API timeout: {endpoint}")
            return None
        except Exception as e:
            logger.error(f"✗ HubSpot API error: {e}")
            return None

    # ==================== INVENTORY (PRODUCTS) METHODS ====================
    
    async def create_product(self, product_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a product in HubSpot
        
        Args:
            product_data: {
                "name": "Product Name",
                "hs_sku": "PROD-001",
                "description": "Product description",
                "price": 2999,
                "hs_cost_of_goods_sold": 1500,
                "hs_product_type": "Physical",
                "category": "T-Shirts",  # Custom property
                "quantity_available": 50  # Custom property
            }
        
        Returns:
            Product ID if successful
        """
        endpoint = "/crm/v3/objects/products"
        
        properties = {
            "name": product_data.get("name"),
            "hs_sku": product_data.get("hs_sku"),
            "description": product_data.get("description", ""),
            "price": product_data.get("price"),
        }
        
        # Add optional properties
        if "hs_cost_of_goods_sold" in product_data:
            properties["hs_cost_of_goods_sold"] = product_data["hs_cost_of_goods_sold"]
        if "hs_product_type" in product_data:
            properties["hs_product_type"] = product_data["hs_product_type"]
        if "category" in product_data:
            properties["category"] = product_data["category"]
        
        data = {"properties": properties}
        
        result = await self._request("POST", endpoint, data)
        
        if result and result.get("id"):
            product_id = result["id"]
            logger.info(f"✓ Created product: {product_id} - {product_data.get('name')}")
            return product_id
        
        return None
    
    async def get_product_by_sku(self, sku: str) -> Optional[Dict]:
        """
        Get product by SKU
        
        Args:
            sku: Product SKU
        
        Returns:
            Product data if found
        """
        endpoint = "/crm/v3/objects/products/search"
        
        search_data = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "hs_sku",
                            "operator": "EQ",
                            "value": sku
                        }
                    ]
                }
            ],
            "properties": ["name", "hs_sku", "description", "price", "hs_cost_of_goods_sold", "category"]
        }
        
        result = await self._request("POST", endpoint, search_data)
        
        if result and result.get("results"):
            product = result["results"][0]
            logger.info(f"✓ Found product: {sku}")
            return product
        
        logger.info(f"✗ Product not found: {sku}")
        return None
    
    async def search_products(self, query: str) -> List[Dict]:
        """
        Search products by name, category, or SKU
        Uses improved fuzzy matching with token-based search and plural handling
        
        Args:
            query: Search query (e.g., "t-shirt", "jeans", "T-Shirts", "Hoodies", "Hoodie Classic Gray")
        
        Returns:
            List of matching products sorted by relevance
        """
        # Get all products and filter client-side for better matching
        all_products = await self.get_all_products()
        
        if not all_products:
            return []
        
        query_lower = query.lower().strip()
        
        # Helper function to normalize text (handle plurals)
        def normalize(text):
            """Remove trailing 's' for plural handling"""
            text = text.lower().strip()
            if text.endswith('s') and len(text) > 3:
                return text[:-1]
            return text
        
        # Tokenize query into words
        query_tokens = [normalize(token) for token in query_lower.split()]
        query_normalized = normalize(query_lower)
        
        # Score each product
        scored_products = []
        
        for product in all_products:
            props = product.get("properties", {})
            name = (props.get("name") or "").lower()
            category = (props.get("category") or "").lower()
            sku = (props.get("hs_sku") or "").lower()
            
            score = 0
            
            # 1. Exact match (highest priority)
            if query_lower == name or query_lower == category or query_lower == sku:
                score = 1000
            
            # 2. Normalized exact match (handle plurals)
            elif query_normalized == normalize(name) or query_normalized == normalize(category):
                score = 900
            
            # 3. Exact substring match
            elif query_lower in name or query_lower in category or query_lower in sku:
                score = 800
            
            # 4. Token-based matching (partial word matches)
            else:
                # Tokenize product name
                name_tokens = [normalize(token) for token in name.split()]
                category_tokens = [normalize(token) for token in category.split()] if category else []
                
                # Count how many query tokens match product tokens
                matches = 0
                for query_token in query_tokens:
                    # Check if query token appears in any product token
                    for name_token in name_tokens:
                        if query_token in name_token or name_token in query_token:
                            matches += 1
                            break
                    else:
                        # Also check category tokens
                        for cat_token in category_tokens:
                            if query_token in cat_token or cat_token in query_token:
                                matches += 1
                                break
                
                # Score based on number of matching tokens
                if matches > 0:
                    score = 100 * matches
            
            # Add product with score if it matched
            if score > 0:
                scored_products.append((score, product))
        
        # Sort by score (highest first) and extract products
        scored_products.sort(key=lambda x: x[0], reverse=True)
        matched_products = [product for score, product in scored_products]
        
        logger.info(f"✓ Found {len(matched_products)} products matching '{query}'")
        return matched_products[:20]  # Limit to 20 results
    
    async def get_all_products(self) -> List[Dict]:
        """
        Get all products from HubSpot inventory
        
        Returns:
            List of all products grouped by category
        """
        endpoint = "/crm/v3/objects/products/search"
        
        search_data = {
            "filterGroups": [],
            "properties": ["name", "hs_sku", "description", "price", "category"],
            "limit": 100
        }
        
        result = await self._request("POST", endpoint, search_data)
        
        if result and result.get("results"):
            logger.info(f"✓ Retrieved {len(result['results'])} products from inventory")
            return result["results"]
        
        return []
    
    async def update_product_stock(self, product_id: str, quantity: int) -> bool:
        """
        Update product stock quantity
        Note: This requires a custom property 'quantity_available' to be created
        
        Args:
            product_id: HubSpot product ID
            quantity: New stock quantity
        
        Returns:
            True if successful
        """
        endpoint = f"/crm/v3/objects/products/{product_id}"
        
        data = {
            "properties": {
                "quantity_available": quantity
            }
        }
        
        result = await self._request("PATCH", endpoint, data)
        
        if result:
            logger.info(f"✓ Updated product {product_id} stock to {quantity}")
            return True
        
        return False

    # ==================== ORDERS METHODS ====================
    
    async def create_order(self, order_data: Dict[str, Any]) -> Optional[str]:
        """
        Create an order in HubSpot
        
        Args:
            order_data: {
                "hs_order_name": "ORD-20250112-001",
                "customer_email": "customer@example.com",
                "customer_phone": "+919876543210",
                "hs_total_price": 2999,
                "hs_payment_status": "PENDING",
                "items": "Product 1 x2, Product 2 x1",
                "shipping_address": "123 Street, City"
            }
        
        Returns:
            Order ID if successful
        """
        endpoint = "/crm/v3/objects/orders"
        
        # Start with required fields + pipeline
        properties = {
            "hs_order_name": order_data.get("hs_order_name"),
            "hs_total_price": order_data.get("hs_total_price", 0),
            "hs_pipeline": "14a2e10e-5471-408a-906e-c51f3b04369e",  # Order Pipeline
            "hs_pipeline_stage": "4b27b500-f031-4927-9811-68a0b525cbae",  # Open stage
        }
        
        # Add all optional properties from Excel
        optional_fields = [
            "hs_payment_status",
            "payment_processing_method",
            "hs_order_note",
            "hs_tags",
            "hs_shipping_address_city",
            "hs_external_order_status",
            "hs_billing_address_email",
            "hs_customer_phone",
            "hs_billing_address_name",
            "hs_external_created_date"
        ]
        
        for field in optional_fields:
            if field in order_data and order_data[field]:
                properties[field] = order_data[field]
        
        # Handle legacy field names for backward compatibility
        if "items" in order_data:
            properties["hs_order_note"] = order_data["items"]
        if "customer_email" in order_data:
            properties["hs_billing_address_email"] = order_data["customer_email"]
        if "customer_phone" in order_data:
            properties["hs_customer_phone"] = order_data["customer_phone"]
        if "customer_name" in order_data:
            properties["hs_billing_address_name"] = order_data["customer_name"]
        
        data = {"properties": properties}
        
        # Debug: Log the exact data being sent to HubSpot
        logger.info(f"📤 Sending order data to HubSpot: {data}")
        
        result = await self._request("POST", endpoint, data)
        
        if result and result.get("id"):
            order_id = result["id"]
            logger.info(f"✓ Created order: {order_id} - {order_data.get('hs_order_name')}")
            
            # Associate with contact if we have customer email
            customer_email = order_data.get("hs_billing_address_email") or order_data.get("customer_email")
            if customer_email:
                contact_id = await self._get_or_create_contact(
                    customer_email,
                    order_data.get("hs_customer_phone") or order_data.get("customer_phone"),
                    order_data.get("hs_billing_address_name") or order_data.get("customer_name")
                )
                
                if contact_id:
                    # Create association from contact to order
                    assoc_endpoint = f"/crm/v3/objects/contacts/{contact_id}/associations/orders/{order_id}/508"
                    assoc_result = await self._request("PUT", assoc_endpoint)
                    if assoc_result:
                        logger.info(f"✓ Associated order {order_id} with contact {contact_id}")
            
            return order_id
        
        return None
    
    async def get_order_by_name(self, order_name: str) -> Optional[Dict]:
        """
        Get order by order name
        
        Args:
            order_name: Order name/ID (e.g., "ORD-20250112-001")
        
        Returns:
            Order data if found
        """
        endpoint = "/crm/v3/objects/orders/search"
        
        search_data = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "hs_order_name",
                            "operator": "EQ",
                            "value": order_name
                        }
                    ]
                }
            ],
            "properties": [
                "hs_order_name",
                "hs_total_price",
                "hs_payment_status",
                "hs_order_note",
                "hs_billing_address_email",
                "hs_shipping_address_city",
                "hs_external_order_status",
                "hs_tags",
                "hs_external_created_date"
            ]
        }
        
        result = await self._request("POST", endpoint, search_data)
        
        if result and result.get("results"):
            order = result["results"][0]
            logger.info(f"✓ Found order: {order_name}")
            return order
        
        logger.info(f"✗ Order not found: {order_name}")
        return None
    
    async def update_order(self, order_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing order in HubSpot
        
        Args:
            order_id: HubSpot order ID (internal ID, not order name)
            update_data: Dictionary of properties to update, e.g.:
                {
                    "hs_order_note": "Cap Baseball Black, Classic Black T-Shirt",
                    "hs_tags": "CAP-BB-BLK, TSH-CLS-BLK-M",
                    "hs_total_price": 848
                }
        
        Returns:
            True if successful
        """
        endpoint = f"/crm/v3/objects/orders/{order_id}"
        
        data = {"properties": update_data}
        
        result = await self._request("PATCH", endpoint, data)
        
        if result:
            logger.info(f"✓ Updated order {order_id}")
            return True
        
        logger.error(f"✗ Failed to update order {order_id}")
        return False
    
    async def get_orders_by_email(self, email: str) -> List[Dict]:
        """
        Get all orders for a customer by email
        
        Args:
            email: Customer email
        
        Returns:
            List of orders
        """
        # First find the contact
        from hubspot_integration import HubSpotClient
        client = HubSpotClient()
        
        # Search for contact by email
        endpoint = "/crm/v3/objects/contacts/search"
        search_data = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "email",
                            "operator": "EQ",
                            "value": email
                        }
                    ]
                }
            ],
            "properties": ["email"]
        }
        
        result = await self._request("POST", endpoint, search_data)
        
        if not result or not result.get("results"):
            logger.info(f"✗ No contact found for email: {email}")
            return []
        
        contact_id = result["results"][0]["id"]
        
        # Get associated orders
        endpoint = f"/crm/v3/objects/contacts/{contact_id}/associations/orders"
        result = await self._request("GET", endpoint)
        
        if result and result.get("results"):
            order_ids = [r["id"] for r in result["results"]]
            logger.info(f"✓ Found {len(order_ids)} orders for {email}")
            
            # Fetch order details
            orders = []
            for order_id in order_ids:
                order_endpoint = f"/crm/v3/objects/orders/{order_id}?properties=hs_order_name,hs_total_price,hs_payment_status,createdate"
                order_result = await self._request("GET", order_endpoint)
                if order_result:
                    orders.append(order_result)
            
            return orders
        
        return []
    
    async def _get_or_create_contact(self, email: str, phone: Optional[str] = None, name: Optional[str] = None) -> Optional[str]:
        """Get existing contact or create new one"""
        from hubspot_integration import HubSpotClient
        client = HubSpotClient()
        
        # Search by email
        endpoint = "/crm/v3/objects/contacts/search"
        search_data = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "email",
                            "operator": "EQ",
                            "value": email
                        }
                    ]
                }
            ]
        }
        
        result = await self._request("POST", endpoint, search_data)
        
        if result and result.get("results"):
            contact_id = result["results"][0]["id"]
            logger.info(f"✓ Found existing contact: {contact_id}")
            return contact_id
        
        # Create new contact
        properties = {"email": email}
        if phone:
            properties["phone"] = phone
        if name:
            properties["firstname"] = name.split()[0] if name else ""
            if len(name.split()) > 1:
                properties["lastname"] = " ".join(name.split()[1:])
        
        endpoint = "/crm/v3/objects/contacts"
        data = {"properties": properties}
        
        result = await self._request("POST", endpoint, data)
        
        if result and result.get("id"):
            contact_id = result["id"]
            logger.info(f"✓ Created contact: {contact_id} for {email}")
            return contact_id
        
        return None


# Global instance
_inventory_orders_manager: Optional[HubSpotInventoryOrdersManager] = None


def get_inventory_orders_manager() -> HubSpotInventoryOrdersManager:
    """Get or create global HubSpotInventoryOrdersManager instance"""
    global _inventory_orders_manager
    if _inventory_orders_manager is None:
        _inventory_orders_manager = HubSpotInventoryOrdersManager()
    return _inventory_orders_manager
