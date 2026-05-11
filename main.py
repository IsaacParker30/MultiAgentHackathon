from config import get_llm_config
from agents import create_agents
from workflow import run_planning_phase, run_execution_phase


def main():
    print("=" * 60)
    print("  ML Workflow Planner — Multi-Agent System")
    print("=" * 60)
    print()

    llm_config = get_llm_config()
    agents = create_agents(llm_config)

    print("Describe your ML task:")
    print("  e.g. 'Fine-tune a sentiment classifier on IMDB using DistilBERT'")
    print()
    task = input("> ").strip()
    if not task:
        print("No task provided. Exiting.")
        return

    print("\n" + "=" * 60)
    print("  PHASE 1: PLANNING")
    print("=" * 60 + "\n")

    plan = run_planning_phase(agents, llm_config, task)

    print("\n" + "=" * 60)
    print("  PLAN COMPLETE")
    print("=" * 60 + "\n")

    proceed = input("Proceed to execution? (y/n): ").strip().lower()
    if proceed == "y":
        print("\n" + "=" * 60)
        print("  PHASE 2: EXECUTION")
        print("=" * 60 + "\n")
        run_execution_phase(agents, llm_config, plan)
    else:
        print("\nPlan saved. Exiting without execution.")

    print("\nDone!")


if __name__ == "__main__":
    main()
