import logging

from langchain_core.messages import HumanMessage, ToolMessage

from src.agents.application.registry import AgentRegistry
from src.config import settings
from src.infrastructure.checkpoint_factory import CheckpointFactory
from src.infrastructure.vector_db import VectorStoreManager
from src.workflow.graph import WorkflowBuilder

logging.basicConfig(level=logging.INFO)

def test_nodes_isolated():
    print("\n" + "=" * 50)
    print("🚀 TESTING NODES")
    print("=" * 50)

    db_manager = VectorStoreManager()
    checkpointer = CheckpointFactory.get_checkpointer()

    raw_base_url = settings.models.get("base_url")
    if not raw_base_url or str(raw_base_url).strip().lower() == "none":
        clean_base_url = None
    else:
        clean_base_url = raw_base_url

    registry = AgentRegistry(
    api_key=settings.token, 
    model_id=settings.models.model_id,
    base_url=clean_base_url
    )    
    builder = WorkflowBuilder(registry, db_manager)

    user_input = "Tengo una reunion con un cliente en Madrid, en la calle Delicias numero 20. Yo estoy en Valencia y la reunión son los días 12 y 13 de Noviembre"
    # user_input = "Tengo reunión en Londres (12-14 Nov). Quiero el mejor hotel y vuelos en Business."
    
    mock_state = {
        "messages": [HumanMessage(content=user_input)],
        "user_role": "Employee",
        "rag_facts": None,
        "planner_facts": None,
        "operator_facts": None
    }

    # ==========================================
    # TEST 0: RETRIEVER
    # ==========================================
    print("\n--- TEST 0: (RAG) ---")
    if db_manager:
        #collection_name="te_policies_compliance"
        collection_name="preferred_vendors"
        try:
            retriever = db_manager.get_retriever(collection_name=collection_name)
            docs = retriever.invoke(mock_state["messages"][0].content)
            print(f"✅ SUCESS: {len(docs)} retrieved from db")
            for i, doc in enumerate(docs[:2]):
                print(f"   Doc {i+1}: {doc.page_content[:150]}...")
        except Exception as e:
            print(f"❌ ERROR RETRIEVER: {e}")
            return
    else:
        print("⚠️ Warning: db_manager is None")

    # ==========================================
    # TEST 1: RESEARCHER NODE
    # ==========================================
    print("\n--- TEST 1: RESEARCHER NODE ---")
    try:
        res1 = builder._researcher_node(mock_state)
        print("✅ Researcher generates RAGFacts:")
        print(res1["rag_facts"].model_dump_json(indent=2))
        mock_state["rag_facts"] = res1["rag_facts"]
    except Exception as e:
        print(f"❌ ERROR RESEARCHER: {e}")
        return

    # ==========================================
    # TEST 2: PLANNER NODE (ReAct)
    # ==========================================
    print("\n--- TEST 2: PLANNER NODE (ReAct) ---")
    try:
        res2 = builder._planner_node(mock_state)
        ai_message = res2["messages"][0]
        
        print("✅ Planner response")
        mock_state["messages"].append(ai_message) 

        if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
            print(f"🛠️ Planner will use {len(ai_message.tool_calls)} tools:")
            
            # --- TEST 2.5: REAL TOOL EXECUTION ---
            print("\n--- TEST 2.5: REAL TOOL EXECUTION ---")
            
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"🔄 Executing: {tool_name} con {tool_args}")
                
                tool_instance = next((t for t in builder.planner_agent.tools if t.name == tool_name), None)                
                if tool_instance:
                    tool_result = tool_instance.invoke(tool_args)
                    print(f"✅ Result of {tool_name}:\n{tool_result}\n")
                    
                    mock_state["messages"].append(
                        ToolMessage(
                            content=str(tool_result), 
                            name=tool_name, 
                            tool_call_id=tool_call["id"]
                        )
                    )
                else:
                    print(f"❌ Tool '{tool_name}' not found in builder.planner_tools")
        else:
            print("⚠️ Warning: Planner didn't use tools, only text:")
            print(ai_message.content)
            
    except Exception as e:
        print(f"❌ ERROR PLANNER: {e}")
        return

    # ==========================================
    # TEST 3: PLANNER EXTRACTOR NODE
    # ==========================================
    print("\n--- TEST 3: PLANNER EXTRACTOR ---")
    try:
        res3 = builder._planner_extractor_node(mock_state)
        print("✅ Extractor generated PlannerFacts:")
        print(res3["planner_facts"].model_dump_json(indent=2))
        
        mock_state["planner_facts"] = res3["planner_facts"]
    except Exception as e:
        print(f"❌ ERROR EXTRACTOR: {e}")
        return

    print("\n" + "="*50)
    print("🎉 COMPLETED TESTS")
    print("="*50)

if __name__ == "__main__":
    test_nodes_isolated()