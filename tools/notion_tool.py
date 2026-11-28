from notion_client import Client
from langchain.tools import tool
from app.config import settings

def get_notion_client(notion_api: str | None = None) -> Client:
    api_key = notion_api or settings.notion_api_key
    if not api_key:
        raise ValueError("Notion API kaliti sozlanmagan")
    return Client(auth=api_key)

def _split_content(content: str) -> list[str]:
    content_splits = [content[i:i+1900] for i in range(0, len(content), 1900)]
    return content_splits

@tool
def save_to_notion(title: str, content: str) -> str:
    """Saves a title and text content as a new subpage in Notion.

    Args:
        title: The title of the Notion page.
        content: The full text content or summary to save.
    """
    if not settings.notion_api_key or not settings.notion_page_id:
        return "Notion API kaliti yoki sahifa Id sozlanmagan"
    try: 
        notion = get_notion_client(settings.notion_api_key)
        chunks = _split_content(content)
        children = [
            {"object": "block", 
             "type": "paragraph", 
             "paragraph": {
                 "rich_text": [{"type": "text", "text": {"content": chunk}}]
                 }
            }
            for chunk in chunks
        ]
        response = notion.pages.create(
            parent={"page_id": settings.notion_page_id},
            properties = {"title": {"title": [{"text": {"content": title}}]}},
            children=children[:100]
        )

        page_id = response["id"]
        for i in range(100, len(children), 100):
            notion.blocks.children.append(
                block_id=page_id,
                children=children[i:i+100]
            )
        page_url = response.get("url", "")

        return f"Successfully saved to Notion: {page_url}"
    
    except Exception as exc:
        return f"Error saving to Notion: {exc}"
    