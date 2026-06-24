import sys
import uuid

from src.workflow.service import AgentOrchestrator

def run_cli() -> None:
    print("✈️  VERITAS CORPORATE TRAVEL - TERMINAL INTERFACE")
    print("Type 'exit' or 'quit' to end the session.")
    print("="*60)

    try:
        orchestrator = AgentOrchestrator()
    except Exception as e:
        print(f"System failed to start: {e}")
        sys.exit(1)

    thread_id = str(uuid.uuid4())
    awaiting_approval = False  # Human-in-the-Loop control flag

    while True:
        try:
            # 1. INPUT ROUTING (Employee vs Manager)
            if awaiting_approval:
                print("\n" + "!"*60)
                print("⚠️  TRIP PENDING MANAGER APPROVAL (Out of Policy / High Cost)")
                print("!"*60)
                choice = input("Action required [A: Approve / R: Reject]: ").strip().upper()
                feedback = input("Manager comment (optional): ").strip()

                if choice == 'A':
                    response = orchestrator.resume_suspended_query(thread_id, approve=True, feedback=feedback)
                else:
                    response = orchestrator.resume_suspended_query(thread_id, approve=False, feedback=feedback)

                print("\n🤖 VERITAS CORPORATE SYSTEM:")
                print(response.get("respuesta", "No response returned."))
                awaiting_approval = False
                continue
            else:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['salir', 'exit', 'quit']:
                    print("👋 Closing system...")
                    break
                    
                if not user_input:
                    continue

            # 2. STREAMING EXECUTION
            print("\n🤖 VERITAS CORPORATE SYSTEM:")
            
            # Use the stream generator instead of process_query
            for event in orchestrator.stream_query(user_input, thread_id):
                event_type = event.get("type")
                
                # UX transparency: show which node is currently running
                if event_type == "node_update":
                    print(f"  [⚙️  Agent '{event.get('node')}' finished task]")
                
                # Streaming: print word-by-word without newline
                elif event_type == "token":
                    print(event.get("content"), end="", flush=True)
                
                # Interruption: graph detected a policy breach and paused
                elif event_type == "interrupted":
                    awaiting_approval = True
                    break  # Exit inner loop to prompt manager input
                
                elif event_type == "error":
                    print(f"\n❌ Execution error: {event.get('content')}")
            
            # Aesthetic newline after response
            print("\n" + "-"*60 if not awaiting_approval else "")
            
        except KeyboardInterrupt: # Catch Ctrl+C
            print("\n👋 Session interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")

if __name__ == "__main__":
    run_cli()