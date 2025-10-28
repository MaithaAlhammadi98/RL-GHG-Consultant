from __future__ import annotations
from typing import Dict, Any, Optional

ACTIONS = ["broad", "legal_only", "financial_only", "company_only"]

def action_to_filter(action: str, company_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    a = (action or "").lower()
    if a == "legal_only":
        # Retrieve technical GHG methodology and API documentation
        return {"source": {"$in": ["2021-API-GHG-Compendium-110921.pdf"]}}
    if a == "financial_only":
        # Retrieve international standards and compliance guidelines
        return {"source": {"$in": ["ISO-14064-1.pdf"]}}
    if a == "company_only":
        # Retrieve regulatory and sustainability reporting standards
        return {"source": {"$in": ["24ru-12-australian-sustainability-reporting-standards-legislation-finalised.pdf"]}}
    # "broad" - no filter, retrieve from all available documents
    return None



