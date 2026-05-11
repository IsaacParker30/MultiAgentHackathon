import os
from datetime import datetime

from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig
from autogen.coding import LocalCommandLineCodeExecutor

TERMINATION_KEYWORD = "APPROVED - READY TO EXECUTE"
EXECUTION_COMPLETE_KEYWORD = "EXECUTION COMPLETE"


def _is_execution_complete(msg: dict) -> bool:
    content = (msg.get("content", "") or "").strip()
    if EXECUTION_COMPLETE_KEYWORD not in content:
        return False
    if "```" in content:
        return False
    return True


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


def run_execution_phase(agents: dict, llm_config: LLMConfig, plan_text: str, task_description: str = "") -> None:
    work_dir = _setup_workflow_dir(task_description or "workflow")

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
        is_termination_msg=_is_execution_complete,
    )

    plan_path = os.path.join(work_dir, "plan.md")
    with open(plan_path, "w") as f:
        f.write(plan_text)

    opening = (
        f"Execute the following approved workflow plan. "
        f"Generate a single complete Python script. "
        f"All output files will be saved to: {work_dir}\n\n"
        f"{plan_text}"
    )

    exec_user.initiate_chat(manager, message=opening)
    print(f"\nWorkflow directory: {work_dir}")


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


def _setup_workflow_dir(task_description: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "_".join(task_description.lower().split()[:5])
    slug = "".join(c if c.isalnum() or c == "_" else "" for c in slug)
    workflow_dir = os.path.join("workflows", f"{timestamp}_{slug}")
    os.makedirs(workflow_dir, exist_ok=True)
    return workflow_dir
