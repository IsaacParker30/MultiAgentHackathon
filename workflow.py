import os
from datetime import datetime

from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig
from autogen.coding import LocalCommandLineCodeExecutor

TERMINATION_KEYWORD = "APPROVED - READY TO EXECUTE"
EXECUTION_COMPLETE_KEYWORD = "EXECUTION COMPLETE"


def run_planning_phase(agents: dict, llm_config: LLMConfig, task_description: str) -> str:
    planner = agents["planner"]
    data_eng = agents["data_engineer"]
    ml_eng = agents["ml_engineer"]
    evaluator = agents["evaluator"]
    user = agents["user_proxy"]

    planning_agents = [user, planner, data_eng, ml_eng, evaluator]

    allowed_transitions = {
        user: [planner],
        planner: [data_eng, ml_eng, evaluator, user],
        data_eng: [planner, ml_eng, evaluator],
        ml_eng: [planner, data_eng, evaluator],
        evaluator: [planner, data_eng, ml_eng],
    }

    group_chat = GroupChat(
        agents=planning_agents,
        messages=[],
        max_round=20,
        speaker_selection_method="auto",
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
        send_introductions=True,
    )

    manager = GroupChatManager(
        name="PlanningManager",
        groupchat=group_chat,
        llm_config=llm_config,
        is_termination_msg=lambda x: TERMINATION_KEYWORD in (x.get("content", "") or ""),
    )

    chat_result = user.initiate_chat(manager, message=task_description)
    return _extract_plan(chat_result)


def run_execution_phase(agents: dict, llm_config: LLMConfig, plan_text: str) -> None:
    work_dir = _setup_output_dir()

    executor_agent = agents["executor"]
    evaluator = agents["evaluator"]

    exec_user = ConversableAgent(
        name="ExecUser",
        system_message="You are the human user. Review generated code and approve execution.",
        human_input_mode="ALWAYS",
        llm_config=False,
        code_execution_config={
            "executor": LocalCommandLineCodeExecutor(work_dir=work_dir),
        },
    )

    execution_agents = [exec_user, executor_agent, evaluator]

    allowed_transitions = {
        executor_agent: [exec_user],
        exec_user: [executor_agent, evaluator],
        evaluator: [exec_user, executor_agent],
    }

    group_chat = GroupChat(
        agents=execution_agents,
        messages=[],
        max_round=30,
        speaker_selection_method="auto",
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
        send_introductions=True,
    )

    manager = GroupChatManager(
        name="ExecutionManager",
        groupchat=group_chat,
        llm_config=llm_config,
        is_termination_msg=lambda x: EXECUTION_COMPLETE_KEYWORD in (x.get("content", "") or ""),
    )

    opening = (
        "Execute the following approved ML workflow plan step by step. "
        "Generate complete Python code for each step, wait for approval "
        "before proceeding to the next.\n\n"
        f"{plan_text}"
    )

    exec_user.initiate_chat(manager, message=opening)
    print(f"\nAll outputs saved to: {work_dir}")


def _extract_plan(chat_result) -> str:
    for msg in reversed(chat_result.chat_history):
        content = msg.get("content", "") or ""
        if "## Workflow Plan" in content:
            return content
    for msg in reversed(chat_result.chat_history):
        content = msg.get("content", "") or ""
        if msg.get("name") != "User" and content:
            return content
    return "No plan found."


def _setup_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join("output", f"run_{timestamp}")
    os.makedirs(work_dir, exist_ok=True)
    return work_dir
