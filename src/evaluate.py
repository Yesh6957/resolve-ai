import json
import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from openai import OpenAI

from pipeline import SupportAgent


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

GOLDEN_PATH = (
    PROCESSED_DIR
    / "golden_dataset.csv"
)

EVAL_RESULTS_PATH = (
    PROCESSED_DIR
    / "evaluation_results.csv"
)


# ============================================================
# ENVIRONMENT
# ============================================================

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_PATH
)


# ============================================================
# LLM7 SETTINGS
# ============================================================

LLM7_BASE_URL = "https://api.llm7.io/v1"

LLM7_API_TOKEN = (
    os.getenv("LLM7_API_TOKEN")
    or os.getenv("LLM7_API_KEY")
)

SUPPORT_MODEL = os.getenv(
    "LLM7_SUPPORT_MODEL",
    "codestral-latest"
)

JUDGE_MODEL = os.getenv(
    "LLM7_JUDGE_MODEL",
    "codestral-latest"
)


# ============================================================
# REQUEST SETTINGS
# ============================================================

REQUEST_DELAY = float(
    os.getenv(
        "LLM7_REQUEST_DELAY",
        "2"
    )
)

MAX_RETRIES = int(
    os.getenv(
        "LLM7_MAX_RETRIES",
        "3"
    )
)

JUDGE_SAMPLE_SIZE = int(
    os.getenv(
        "JUDGE_SAMPLE_SIZE",
        "10"
    )
)


# ============================================================
# CUSTOM EXCEPTION
# ============================================================

class QuotaExceededError(Exception):
    """Raised when LLM7 quota or rate limit is exhausted."""
    pass


# ============================================================
# EVALUATOR
# ============================================================

class Evaluator:

    def __init__(self):

        print("=" * 60)
        print("🧪 INITIALIZING EVALUATION HARNESS")
        print("=" * 60)

        # ----------------------------------------------------
        # Check API token
        # ----------------------------------------------------

        if not LLM7_API_TOKEN:

            raise ValueError(
                "\n❌ LLM7 API token not found.\n\n"
                f"Expected .env file:\n{ENV_PATH}\n\n"
                "Add:\n"
                "LLM7_API_TOKEN=your_api_key\n"
            )

        # ----------------------------------------------------
        # Create LLM7 client
        # ----------------------------------------------------

        self.client = OpenAI(
            api_key=LLM7_API_TOKEN,
            base_url=LLM7_BASE_URL
        )

        # ----------------------------------------------------
        # Initialize SupportAgent
        # ----------------------------------------------------

        print(
            "\n🔧 Initializing SupportAgent..."
        )

        self.agent = SupportAgent()

        # Force the evaluator and agent to use
        # the same API client/model.

        self.agent.client = self.client
        self.agent.model = SUPPORT_MODEL

        self.judge_model = JUDGE_MODEL

        # ----------------------------------------------------
        # Configuration output
        # ----------------------------------------------------

        print(
            f"\n🤖 Support model: "
            f"{SUPPORT_MODEL}"
        )

        print(
            f"🤖 Judge model: "
            f"{JUDGE_MODEL}"
        )

        print(
            f"🌐 API: "
            f"{LLM7_BASE_URL}"
        )

        print(
            f"⏱️ Request delay: "
            f"{REQUEST_DELAY}s"
        )

        print(
            f"🔁 Max retries: "
            f"{MAX_RETRIES}"
        )

        print(
            f"🤖 Judge sample size: "
            f"{JUDGE_SAMPLE_SIZE}"
        )


    # ========================================================
    # ERROR DETECTION
    # ========================================================

    @staticmethod
    def is_quota_error(error):

        text = str(error).upper()

        return (
            "429" in text
            or "RATE LIMIT" in text
            or "RATE_LIMIT" in text
            or "TOO MANY REQUESTS" in text
            or "QUOTA" in text
            or "INSUFFICIENT QUOTA" in text
            or "LIMIT EXCEEDED" in text
            or "TPM" in text
            or "RPM" in text
        )


    @staticmethod
    def is_model_error(error):

        text = str(error).lower()

        return (
            "model_unavailable" in text
            or "model is currently unavailable" in text
            or "model unavailable" in text
        )


    # ========================================================
    # BOOLEAN NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_bool(value):

        if isinstance(value, bool):
            return value

        if value is None:
            return False

        text = (
            str(value)
            .strip()
            .lower()
        )

        if text in {
            "true",
            "1",
            "yes",
            "y",
            "t"
        }:
            return True

        if text in {
            "false",
            "0",
            "no",
            "n",
            "f"
        }:
            return False

        return False


    # ========================================================
    # PREPARE JUDGE COLUMNS
    # ========================================================

    @staticmethod
    def prepare_judge_columns(df):

        columns = [
            "judge_tone",
            "judge_grounding",
            "judge_reasoning"
        ]

        for column in columns:

            if column not in df.columns:
                df[column] = None

            df[column] = (
                df[column]
                .astype("object")
            )

        return df


    # ========================================================
    # LLM AS A JUDGE
    # ========================================================

    def llm_as_a_judge(
        self,
        customer_text,
        drafted_reply
    ):

        prompt = f"""
You are a strict QA evaluator for an Apple customer-support AI agent.

Customer message:
"{customer_text}"

AI-generated reply:
"{drafted_reply}"

Evaluate the AI reply on two dimensions.

1. TONE

Evaluate whether the response is:

- empathetic
- professional
- concise
- appropriate for customer support
- not robotic or dismissive

2. GROUNDING

Evaluate whether the response:

- addresses the customer's actual issue
- gives a realistic next step
- avoids unsupported policies
- avoids invented claims
- avoids invented URLs
- does not pretend to have performed actions it cannot perform

Scoring:

5 = excellent
4 = good
3 = acceptable
2 = poor
1 = very poor

Return ONLY valid JSON:

{{
    "score_tone": 1,
    "score_grounding": 1,
    "reasoning": "Brief explanation"
}}
"""

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            try:

                response = (
                    self.client
                    .chat
                    .completions
                    .create(
                        model=self.judge_model,

                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a strict "
                                    "support QA evaluator. "
                                    "Return only valid JSON."
                                )
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],

                        temperature=0,

                        max_tokens=300,

                        response_format={
                            "type": "json_object"
                        }
                    )
                )

                if not response.choices:

                    raise ValueError(
                        "Judge returned no choices."
                    )

                response_text = (
                    response
                    .choices[0]
                    .message
                    .content
                    or ""
                ).strip()

                if not response_text:

                    raise ValueError(
                        "Judge returned empty response."
                    )

                # Remove markdown fences.

                if response_text.startswith("```"):

                    response_text = (
                        response_text
                        .replace(
                            "```json",
                            ""
                        )
                        .replace(
                            "```JSON",
                            ""
                        )
                        .replace(
                            "```",
                            ""
                        )
                        .strip()
                    )

                data = json.loads(
                    response_text
                )

                tone = data.get(
                    "score_tone"
                )

                grounding = data.get(
                    "score_grounding"
                )

                try:
                    tone = int(tone)
                except (
                    TypeError,
                    ValueError
                ):
                    tone = None

                try:
                    grounding = int(
                        grounding
                    )
                except (
                    TypeError,
                    ValueError
                ):
                    grounding = None

                if tone not in {
                    1, 2, 3, 4, 5
                }:
                    tone = None

                if grounding not in {
                    1, 2, 3, 4, 5
                }:
                    grounding = None

                return {
                    "score_tone": tone,
                    "score_grounding": grounding,
                    "reasoning": data.get(
                        "reasoning"
                    )
                }

            except Exception as e:

                print(
                    f"\n⚠️ Judge error "
                    f"(attempt "
                    f"{attempt}/"
                    f"{MAX_RETRIES})"
                )

                print(e)

                if self.is_model_error(e):

                    raise ValueError(
                        "\n❌ Judge model unavailable:\n"
                        f"{self.judge_model}\n\n"
                        "Check .env:\n"
                        "LLM7_JUDGE_MODEL=codestral-latest"
                    ) from e

                if self.is_quota_error(e):

                    raise QuotaExceededError(
                        "LLM7 quota/rate limit "
                        "exceeded."
                    ) from e

                if attempt < MAX_RETRIES:

                    wait_time = (
                        REQUEST_DELAY
                        * attempt
                    )

                    print(
                        f"⏳ Retrying in "
                        f"{wait_time:.1f}s..."
                    )

                    time.sleep(
                        wait_time
                    )

        return {
            "score_tone": None,
            "score_grounding": None,
            "reasoning": None
        }


    # ========================================================
    # RUN EVALUATION
    # ========================================================

    def run_evaluation(self):

        print(
            "\n🚀 Loading Golden Dataset..."
        )

        # ----------------------------------------------------
        # Check golden dataset
        # ----------------------------------------------------

        if not GOLDEN_PATH.exists():

            raise FileNotFoundError(
                "\n❌ Golden dataset not found:\n"
                f"{GOLDEN_PATH}"
            )

        df = pd.read_csv(
            GOLDEN_PATH
        )

        print(
            f"✅ Loaded {len(df)} golden examples."
        )

        # ----------------------------------------------------
        # Validate columns
        # ----------------------------------------------------

        required_columns = [
            "customer_text",
            "true_intent",
            "should_escalate"
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                "\n❌ Golden dataset missing columns:\n"
                f"{missing}\n\n"
                f"Available columns:\n"
                f"{list(df.columns)}"
            )

        # ----------------------------------------------------
        # Normalize ground truth escalation
        # ----------------------------------------------------

        df["should_escalate"] = (
            df["should_escalate"]
            .apply(self.normalize_bool)
            .astype(bool)
        )

        # ----------------------------------------------------
        # Existing results
        # ----------------------------------------------------

        if EVAL_RESULTS_PATH.exists():

            previous_df = pd.read_csv(
                EVAL_RESULTS_PATH
            )

            previous_df = (
                self.prepare_judge_columns(
                    previous_df
                )
            )

            print(
                f"📂 Existing results found: "
                f"{len(previous_df)} rows."
            )

        else:

            previous_df = pd.DataFrame()

            print(
                "📂 No previous evaluation found."
            )

        # ----------------------------------------------------
        # Determine IDs
        # ----------------------------------------------------

        completed_ids = set()

        if (
            not previous_df.empty
            and "thread_id"
            in previous_df.columns
        ):

            completed_ids = set(
                previous_df[
                    "thread_id"
                ]
                .astype(str)
            )

        # ----------------------------------------------------
        # Determine rows to evaluate
        # ----------------------------------------------------

        rows_to_process = []

        for idx, row in df.iterrows():

            if "thread_id" in row.index:

                thread_id = str(
                    row["thread_id"]
                )

            else:

                thread_id = (
                    f"golden_{idx}"
                )

            if thread_id not in completed_ids:

                rows_to_process.append(
                    (
                        idx,
                        row,
                        thread_id
                    )
                )

        print(
            f"\n📊 Total golden examples: "
            f"{len(df)}"
        )

        print(
            f"✅ Already evaluated: "
            f"{len(completed_ids)}"
        )

        print(
            f"⏳ Remaining: "
            f"{len(rows_to_process)}"
        )

        # ----------------------------------------------------
        # Nothing remaining
        # ----------------------------------------------------

        if not rows_to_process:

            print(
                "\n🎉 All golden examples "
                "already evaluated."
            )

            if not previous_df.empty:

                self.run_judge_sample(
                    previous_df
                )

                final_df = pd.read_csv(
                    EVAL_RESULTS_PATH
                )

                self.print_metrics(
                    final_df
                )

            return

        # ----------------------------------------------------
        # Evaluation loop
        # ----------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "🚀 RUNNING SUPPORT AGENT"
        )

        print(
            "=" * 60
        )

        new_results = []

        quota_exceeded = False

        progress = tqdm(
            rows_to_process,
            total=len(rows_to_process),
            desc="Evaluating"
        )

        for position, (
            idx,
            row,
            thread_id
        ) in enumerate(
            progress,
            start=1
        ):

            customer_text = str(
                row["customer_text"]
            ).strip()

            print(
                f"\n{'-' * 60}"
            )

            print(
                f"📝 Example "
                f"{position}/"
                f"{len(rows_to_process)}"
            )

            print(
                f"👤 {customer_text}"
            )

            # ------------------------------------------------
            # Run actual agent
            # ------------------------------------------------

            try:

                output = (
                    self.agent
                    .process_tweet(
                        customer_text
                    )
                )

            except Exception as e:

                print(
                    "\n❌ Agent error:"
                )

                print(e)

                if self.is_model_error(e):

                    print(
                        "\n🛑 Model unavailable."
                    )

                    break

                if self.is_quota_error(e):

                    print(
                        "\n🚫 API quota/rate limit "
                        "reached."
                    )

                    quota_exceeded = True

                    break

                print(
                    "\n⚠️ Skipping this example."
                )

                continue

            # ------------------------------------------------
            # Extract analysis
            # ------------------------------------------------

            analysis = output.get(
                "analysis",
                {}
            )

            pred_intent = (
                analysis.get(
                    "intent"
                )
            )

            confidence = (
                analysis.get(
                    "confidence_score"
                )
            )

            try:

                confidence = float(
                    confidence
                )

            except (
                TypeError,
                ValueError
            ):

                confidence = None

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Use the agent's FINAL status for escalation.
            #
            # The pipeline escalates when:
            #
            # should_escalate == True
            #
            # OR
            #
            # confidence < 0.6
            #
            # Therefore status is the real prediction.
            # ------------------------------------------------

            agent_status = output.get(
                "status"
            )

            pred_escalate = (
                agent_status == "escalated"
            )

            # ------------------------------------------------
            # Get reply
            # ------------------------------------------------

            drafted_reply = output.get(
                "reply"
            )

            # ------------------------------------------------
            # Retrieved examples
            # ------------------------------------------------

            retrieved_examples = output.get(
                "retrieved_examples",
                0
            )

            # ------------------------------------------------
            # Build result
            # ------------------------------------------------

            result = {

                "thread_id": thread_id,

                "customer_text": customer_text,

                "true_intent": str(
                    row["true_intent"]
                ).strip(),

                "pred_intent": pred_intent,

                "confidence_score": confidence,

                "true_escalate": (
                    self.normalize_bool(
                        row["should_escalate"]
                    )
                ),

                "pred_escalate": pred_escalate,

                "agent_status": agent_status,

                "escalation_reason": (
                    analysis.get(
                        "escalation_reason",
                        ""
                    )
                ),

                "drafted_reply": drafted_reply,

                "retrieved_examples": (
                    retrieved_examples
                ),

                "judge_tone": None,

                "judge_grounding": None,

                "judge_reasoning": None
            }

            new_results.append(
                result
            )

            # ------------------------------------------------
            # Save immediately
            # ------------------------------------------------

            new_df = pd.DataFrame(
                new_results
            )

            new_df = (
                self.prepare_judge_columns(
                    new_df
                )
            )

            if previous_df.empty:

                all_results = (
                    new_df.copy()
                )

            else:

                all_results = pd.concat(
                    [
                        previous_df,
                        new_df
                    ],
                    ignore_index=True
                )

            all_results = (
                self.prepare_judge_columns(
                    all_results
                )
            )

            all_results.to_csv(
                EVAL_RESULTS_PATH,
                index=False
            )

            print(
                f"💾 Saved "
                f"{len(all_results)} "
                f"evaluation rows."
            )

            # ------------------------------------------------
            # Wait before next request
            # ------------------------------------------------

            if position < len(
                rows_to_process
            ):

                time.sleep(
                    REQUEST_DELAY
                )

        # ----------------------------------------------------
        # Reload results
        # ----------------------------------------------------

        if EVAL_RESULTS_PATH.exists():

            results_df = pd.read_csv(
                EVAL_RESULTS_PATH
            )

        else:

            results_df = pd.DataFrame(
                new_results
            )

        results_df = (
            self.prepare_judge_columns(
                results_df
            )
        )

        # ----------------------------------------------------
        # Quota handling
        # ----------------------------------------------------

        if quota_exceeded:

            print(
                "\n" + "=" * 60
            )

            print(
                "🛑 EVALUATION PAUSED"
            )

            print(
                "=" * 60
            )

            print(
                f"\nSaved rows: "
                f"{len(results_df)}"
            )

            print(
                f"File:\n"
                f"{EVAL_RESULTS_PATH}"
            )

            print(
                "\nRun the same command again "
                "when the API limit is available."
            )

            print(
                "Previously completed rows "
                "will be skipped."
            )

            self.print_metrics(
                results_df
            )

            return

        # ----------------------------------------------------
        # Run LLM judge
        # ----------------------------------------------------

        self.run_judge_sample(
            results_df
        )

        # ----------------------------------------------------
        # Reload after judge
        # ----------------------------------------------------

        if EVAL_RESULTS_PATH.exists():

            results_df = pd.read_csv(
                EVAL_RESULTS_PATH
            )

        # ----------------------------------------------------
        # Final metrics
        # ----------------------------------------------------

        self.print_metrics(
            results_df
        )


    # ========================================================
    # LLM JUDGE SAMPLE
    # ========================================================

    def run_judge_sample(
        self,
        df
    ):

        if df.empty:

            print(
                "\n⚠️ No evaluation rows available "
                "for judging."
            )

            return

        df = (
            self.prepare_judge_columns(
                df
            )
        )

        # ----------------------------------------------------
        # Only auto-handled replies can be judged
        # ----------------------------------------------------

        eligible = df[
            df["drafted_reply"].notna()
            &
            (
                df["drafted_reply"]
                .astype(str)
                .str.strip()
                .str.len()
                > 0
            )
        ].copy()

        if eligible.empty:

            print(
                "\n⚠️ No generated replies "
                "available for LLM judging."
            )

            df.to_csv(
                EVAL_RESULTS_PATH,
                index=False
            )

            return

        # ----------------------------------------------------
        # Only unjudged examples
        # ----------------------------------------------------

        unjudged = eligible[
            eligible["judge_tone"].isna()
        ]

        if unjudged.empty:

            print(
                "\n✅ All eligible replies "
                "already have judge scores."
            )

            return

        # ----------------------------------------------------
        # Deterministic sample
        # ----------------------------------------------------

        sample_size = min(
            JUDGE_SAMPLE_SIZE,
            len(unjudged)
        )

        sample = unjudged.sample(
            n=sample_size,
            random_state=42
        )

        print(
            "\n" + "=" * 60
        )

        print(
            f"🤖 LLM JUDGE: "
            f"{sample_size} replies"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Run judge
        # ----------------------------------------------------

        for count, (
            idx,
            row
        ) in enumerate(
            sample.iterrows(),
            start=1
        ):

            print(
                f"\n🤖 Judge "
                f"{count}/"
                f"{sample_size}"
            )

            try:

                scores = (
                    self.llm_as_a_judge(
                        row["customer_text"],
                        row["drafted_reply"]
                    )
                )

            except QuotaExceededError:

                print(
                    "\n🚫 Judge quota/rate "
                    "limit exceeded."
                )

                break

            except ValueError as e:

                print(
                    "\n❌ Judge error:"
                )

                print(e)

                break

            # ------------------------------------------------
            # Store scores
            # ------------------------------------------------

            df.at[
                idx,
                "judge_tone"
            ] = scores.get(
                "score_tone"
            )

            df.at[
                idx,
                "judge_grounding"
            ] = scores.get(
                "score_grounding"
            )

            df.at[
                idx,
                "judge_reasoning"
            ] = scores.get(
                "reasoning"
            )

            # ------------------------------------------------
            # Save immediately
            # ------------------------------------------------

            df.to_csv(
                EVAL_RESULTS_PATH,
                index=False
            )

            print(
                f"Tone: "
                f"{scores.get('score_tone')}/5"
            )

            print(
                f"Grounding: "
                f"{scores.get('score_grounding')}/5"
            )

            print(
                "💾 Judge result saved."
            )

            if count < sample_size:

                time.sleep(
                    REQUEST_DELAY
                )

        # ----------------------------------------------------
        # Final save
        # ----------------------------------------------------

        df.to_csv(
            EVAL_RESULTS_PATH,
            index=False
        )

        print(
            f"\n💾 Judge results saved to:"
        )

        print(
            EVAL_RESULTS_PATH
        )


    # ========================================================
    # PRINT METRICS
    # ========================================================

    def print_metrics(
        self,
        df
    ):

        print(
            "\n" + "=" * 60
        )

        print(
            "📊 FINAL EVALUATION RESULTS"
        )

        print(
            "=" * 60
        )

        if df.empty:

            print(
                "\n⚠️ No results."
            )

            return

        # ====================================================
        # DATASET INFORMATION
        # ====================================================

        print(
            "\n📦 DATASET"
        )

        print(
            f"Total rows: "
            f"{len(df)}"
        )

        if "pred_intent" in df.columns:

            completed = df[
                df["pred_intent"].notna()
            ]

            print(
                f"Completed evaluations: "
                f"{len(completed)}/"
                f"{len(df)}"
            )

        # ====================================================
        # INTENT METRICS
        # ====================================================

        if (
            "true_intent" in df.columns
            and "pred_intent" in df.columns
        ):

            valid = df[
                df["true_intent"].notna()
                &
                df["pred_intent"].notna()
            ].copy()

            if not valid.empty:

                intent_accuracy = (
                    accuracy_score(
                        valid["true_intent"],
                        valid["pred_intent"]
                    )
                )

                print(
                    "\n" + "=" * 60
                )

                print(
                    "🎯 INTENT CLASSIFICATION"
                )

                print(
                    "=" * 60
                )

                print(
                    f"\nAccuracy: "
                    f"{intent_accuracy:.2%}"
                )

                print(
                    "\nPer-intent performance:"
                )

                print(
                    classification_report(
                        valid["true_intent"],
                        valid["pred_intent"],
                        zero_division=0
                    )
                )

                labels = sorted(
                    set(
                        valid["true_intent"]
                    )
                    |
                    set(
                        valid["pred_intent"]
                    )
                )

                cm = confusion_matrix(
                    valid["true_intent"],
                    valid["pred_intent"],
                    labels=labels
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=[
                        f"TRUE: {x}"
                        for x in labels
                    ],
                    columns=[
                        f"PRED: {x}"
                        for x in labels
                    ]
                )

                print(
                    "🔍 Confusion Matrix:"
                )

                print(
                    cm_df.to_string()
                )

        # ====================================================
        # ESCALATION METRICS
        # ====================================================

        if (
            "true_escalate" in df.columns
            and "pred_escalate" in df.columns
        ):

            valid = df[
                df["true_escalate"].notna()
                &
                df["pred_escalate"].notna()
            ].copy()

            if not valid.empty:

                true_values = (
                    valid["true_escalate"]
                    .apply(
                        self.normalize_bool
                    )
                    .astype(bool)
                )

                pred_values = (
                    valid["pred_escalate"]
                    .apply(
                        self.normalize_bool
                    )
                    .astype(bool)
                )

                accuracy = accuracy_score(
                    true_values,
                    pred_values
                )

                precision = precision_score(
                    true_values,
                    pred_values,
                    zero_division=0
                )

                recall = recall_score(
                    true_values,
                    pred_values,
                    zero_division=0
                )

                f1 = f1_score(
                    true_values,
                    pred_values,
                    zero_division=0
                )

                print(
                    "\n" + "=" * 60
                )

                print(
                    "🚨 ESCALATION"
                )

                print(
                    "=" * 60
                )

                print(
                    f"\nAccuracy: "
                    f"{accuracy:.2%}"
                )

                print(
                    f"Precision: "
                    f"{precision:.2%}"
                )

                print(
                    f"Recall: "
                    f"{recall:.2%}"
                )

                print(
                    f"F1: "
                    f"{f1:.2%}"
                )

                print(
                    "\nConfusion matrix:"
                )

                escalation_cm = confusion_matrix(
                    true_values,
                    pred_values,
                    labels=[False, True]
                )

                escalation_cm_df = pd.DataFrame(
                    escalation_cm,
                    index=[
                        "TRUE: False",
                        "TRUE: True"
                    ],
                    columns=[
                        "PRED: False",
                        "PRED: True"
                    ]
                )

                print(
                    escalation_cm_df.to_string()
                )

        # ====================================================
        # REPLY GENERATION
        # ====================================================

        if "drafted_reply" in df.columns:

            replies = df[
                df["drafted_reply"].notna()
                &
                (
                    df["drafted_reply"]
                    .astype(str)
                    .str.strip()
                    .str.len()
                    > 0
                )
            ]

            reply_rate = (
                len(replies)
                / len(df)
            )

            print(
                "\n" + "=" * 60
            )

            print(
                "💬 REPLY GENERATION"
            )

            print(
                "=" * 60
            )

            print(
                f"\nGenerated replies: "
                f"{len(replies)}/"
                f"{len(df)}"
            )

            print(
                f"Generation rate: "
                f"{reply_rate:.2%}"
            )

        # ====================================================
        # RETRIEVAL
        # ====================================================

        if "retrieved_examples" in df.columns:

            retrieval = pd.to_numeric(
                df["retrieved_examples"],
                errors="coerce"
            )

            retrieval_valid = (
                retrieval.dropna()
            )

            if not retrieval_valid.empty:

                print(
                    "\n📚 Average retrieved "
                    f"examples: "
                    f"{retrieval_valid.mean():.2f}"
                )

        # ====================================================
        # LLM JUDGE
        # ====================================================

        tone_values = pd.to_numeric(
            df["judge_tone"],
            errors="coerce"
        )

        grounding_values = pd.to_numeric(
            df["judge_grounding"],
            errors="coerce"
        )

        valid_tone = (
            tone_values.dropna()
        )

        valid_grounding = (
            grounding_values.dropna()
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "🤖 LLM-AS-A-JUDGE"
        )

        print(
            "=" * 60
        )

        if not valid_tone.empty:

            print(
                f"\nAverage Tone: "
                f"{valid_tone.mean():.2f}/5"
            )

            print(
                f"Tone samples: "
                f"{len(valid_tone)}"
            )

        else:

            print(
                "\nAverage Tone: "
                "No scores"
            )

        if not valid_grounding.empty:

            print(
                f"Average Grounding: "
                f"{valid_grounding.mean():.2f}/5"
            )

            print(
                f"Grounding samples: "
                f"{len(valid_grounding)}"
            )

        else:

            print(
                "Average Grounding: "
                "No scores"
            )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        print(
            "\n" + "=" * 60
        )

        print(
            "🏁 EVALUATION COMPLETE"
        )

        print(
            "=" * 60
        )

        print(
            f"\n📁 Results:"
        )

        print(
            EVAL_RESULTS_PATH
        )

        print(
            "=" * 60
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        evaluator = Evaluator()

        evaluator.run_evaluation()

    except KeyboardInterrupt:

        print(
            "\n\n🛑 Evaluation interrupted."
        )

        print(
            "💾 Any previously saved "
            "results are safe."
        )

    except Exception as e:

        print(
            "\n❌ FATAL ERROR"
        )

        print(e)
