# Requirements:
# pip install langchain langchain-openai langchain-community langgraph openai beautifulsoup4 requests
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import requests
import json

OPENAI_API_KEY="YOUR_API_KEY"

class MallOfAmericaFinder:
    def __init__(self):
        self.base_url = "https://www.moaapi.net/tenants.php"
        self.stores_cache = None
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY
        )

        self.tools = [
            Tool(name="GetStores", func=self.get_relevant_stores,
                 description="Gets store information based on the query"),
            Tool(name="GetHours", func=self.get_store_hours,
                 description="Gets current store hours")
        ]

        self.agent = create_react_agent(self.llm, tools=self.tools)
        self.checkpointer = MemorySaver()
    
    def fetch_all_stores(self) -> List[Dict]:
        if self.stores_cache:
            return self.stores_cache
        
        try:
            response = requests.get(self.base_url, timeout=10)
            response.raise_for_status()
            self.stores_cache = response.json()
        except Exception as e:
            print(f"Error fetching stores: {e}")
            return []

    def clean_store_data(self, store: Dict) -> Dict:
        return {
            'name': store.get('name', ''),
            'categories': [cat['name'] for cat in store.get('categories', [])],
            'level': store.get('level', ''),
            'location': store.get('location', {}).get('unit_number', ''),
            'type': [t['name'] for t in store.get('type', [])],
            'hours': {
                'regular': store.get('hours', {}).get('regular', []),
                'today': store.get('hours', {}).get('today', {})
            },
            'status': store.get('status', {}).get('name', '')
        }
    
    def get_relevant_stores(self, query: str) -> str:
        all_stores = self.fetch_all_stores()
        query_terms = set(query.lower().split())

        relevant_stores = []
        for store in all_stores:
            store_text = json.dumps([
                store.get('name', ''),
                store.get('categories',[]),
                store.get('type',[])
            ]).lower()

            relevance = sum(term in store_text for term in query_terms)
            if relevance > 0:
                clean_store = self.clean_store_data(store)
                clean_store['relevance'] = relevance
                relevant_stores.append(clean_store)
        top_stores = sorted(relevant_stores, key = lambda x: x['relevance'], reverse=True)[:3]
        return json.dumps(top_stores, indent=2)
    
    def get_store_hours(self, store_name: str) -> str:
        stores = self.fetch_all_stores()
        for store in stores: 
            if store['name'].lower() == store_name.lower():
                hours = store.get('hours', {})
                return json.dumps(hours, indent=2)
        return 'Store not found'

    def find_store(self, query: str) -> str:
        try:
            system_message = SystemMessage(content=f"""You are ByteGuide, a friendly and knowledgeable Mall Guide at Mall of America.
                Style and Personality:
                - Professional yet approachable
                - Young and fresh in tone
                - Use current slang appropriately but professionally
                - Add relevant emojis to make responses engaging
                - Format responses clearly with sections and bullet points
                - Use emojis in all your answers

                Response Guidelines:
                1. Be concise but thorough
                2. Start with the most relevant information
                3. Include practical details (level, location, hours)
                4. Add helpful tips when appropriate
                5. Keep a positive, upbeat tone

                Examples of your style:
                - "Got it! Let me help you find that perfect spot! ✨"
                - "Here are some awesome options I found for you 🎯"
                - "Pro tip: This store is right next to the food court!"

                Remember: You're a helpful local friend guiding visitors through the mall."""
            )

            human_message = HumanMessage(content=f"""
                Find stores in Mall of America matching: {query}

                Provide:
                1. Store names and locations
                2. Current operating hours
                3. Best parking suggestions
                4. Any helpful tips or recommendations
                """
            )

            messages = [system_message, human_message]

            response = self.agent.invoke(
                {"messages": messages},
                config = {"configurable":{"thread_id":hash(query)}}
            )

            
            return f"""
            🏬 Mall of America Results
            ========================
            Search: "{query}"
            
            {response['messages'][-1].content}
            
            ℹ️ Visit information desk for directions
            """
            
        except Exception as e:
            return f"⚠️ Search error: {str(e)}" 

if __name__ == "__main__":
    finder = MallOfAmericaFinder()

    queries = [
        "Hi I'm looking for a hot coffee",
        "Help me, I'm looking for kids clothes",
        "Recommend me attractions for families"
    ]

    for query in queries:
        print(f"\n🔍 Searching for: {query}")
        result = finder.find_store(query)
        print(result)
        print("-" * 50)
    