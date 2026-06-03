# Agent Workforce Governance - Pitch

*Working title: the platform that manages your AI agents like a team, not just a leaderboard.*

## The Problem

Companies are no longer running one AI assistant - they are deploying **fleets of agents** (different models, prompts, and tools) across many tasks. But they manage that fleet **blind**.

Today's evaluation platforms hand you a **score** and stop there. A number on a dashboard does not answer the questions an engineering or operations manager actually has to make every day:

- We have five agents that can answer support tickets. **Which one should take this next ticket?**
- This agent is right 95% of the time on refunds but 60% on billing disputes. **How much should we let it act on its own?**
- A model was swapped last week. **Did any of our agents quietly get worse?**
- An auditor asks why an agent took an action. **Where is the record, the policy, and the approval?**

The result: the wrong agent lands on the wrong task, agents are given too much (or too little) autonomy, performance regressions go unnoticed, and there is no accountable system of record. As fleets grow, this becomes an operational and compliance liability - not just a quality nuisance.

## Why Now?

- **Agents went to production.** In the last 18 months multiagent frameworks (LangGraph, CrewAI, AutoGen) moved from demos into real workflows. Fleets are now big enough that managing them by hand breaks down.
- **Model and config sprawl.** There are dozens of viable models and prompt configurations, and performance varies sharply by task type. Picking the right one per task is now a real, recurring decision - not a one-time setup.
- **Cost and reliability pressure.** Teams need to route work to the *cheapest agent that is good enough*, and to know which agents are trustworthy enough to run unattended.
- **Governance is arriving.** Regulation and internal risk policies (e.g. the EU AI Act's push for human oversight and audit trails) increasingly demand that automated decisions be explainable, gated, and logged.
- **The tooling gap is exposed.** Eval/observability tools have matured at *measuring* quality, but nobody owns the **decision layer** that turns those measurements into who-does-what, how-much-autonomy, and policy enforcement.

The measurement problem is largely solved. The **management** problem is wide open - and that is exactly the moment to build for it.

## Our Solution

An **Agent Workforce Governance platform**. Underneath it is a rigorous **multiagent LLM-as-judge eval engine** (built on LangGraph) that continuously scores every agent in the fleet across multiple criteria - correctness, faithfulness, completeness, coherence, and safety - and writes the results to a **performance store** over time.

We trust that engine because we hold it to a hard standard: its verdicts are validated against **human gold labels** (target: >= 80% agreement, Cohen's kappa >= 0.6). It is a judge we can prove is fair before we let it make calls.

On top of that performance history sits the part competitors skip - a **governance layer** that turns scores into decisions.

### The 4 Questions Other Platforms Skip

1. **Which agent should get the next task?**
   A performance-aware **router** picks the best agent for each incoming task using historical competence on that task type, plus reliability, cost, and latency - instead of a static default.

2. **How much autonomy should each agent have?**
   An **autonomy calibrator** maps each agent's measured reliability and risk to explicit tiers - *full auto -> auto with spot-check -> human-in-the-loop -> blocked* - so trust is earned from evidence, not guessed.

3. **How is each agent actually performing over time?**
   Automated **performance reviews**: per-agent scorecards showing strengths and weaknesses by criterion and task type, trends, and regressions - so a silent drop after a model swap gets caught immediately.

4. **How do we govern the fleet?**
   A **policy and governance engine** with an **audit log** that gates actions (low-autonomy agents require approval, safety-critical tasks require human review), enforces escalation paths, and records every decision for accountability and compliance.

### Why it wins

Eval platforms tell you *how good* an agent is. We tell you **what to do about it** - who works, who is trusted, who is improving or slipping, and who needs a human in the loop - with the receipts to prove it. We do not replace your agents; we **manage** them.
