import os
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import chromadb
from openai import OpenAI


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RAG_CSV_PATH = PROCESSED_DIR / "rag_knowledge_base.csv"
CHROMA_DB_PATH = PROCESSED_DIR / "chroma_db"

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


# ============================================================
# LLM7 SETTINGS
# ============================================================

LLM7_BASE_URL = "https://api.llm7.io/v1"

DEFAULT_LLM7_MODEL = "codestral-latest"

REQUEST_TIMEOUT = 60

MAX_RETRIES = 3


# ============================================================
# INTENT DEFINITIONS
# ============================================================

VALID_INTENTS = {
    "account_security",
    "hardware_repair",
    "software_troubleshooting",
    "billing_purchases",
    "device_connectivity",
    "general_inquiry",
    "other",
}


INTENT_DEFINITIONS = """
INTENT DEFINITIONS

account_security:
Problems involving Apple ID, passwords, account access,
login, locked accounts, verification, authentication,
security, compromised accounts, or sign-in.

hardware_repair:
Physical device problems or suspected physical hardware failure.
Examples: cracked screen, broken buttons, damaged device,
water damage, swollen battery, camera hardware failure,
speaker hardware failure, charging-port damage.

software_troubleshooting:
Problems caused by software, iOS, apps, settings, crashes,
bugs, glitches, updates, installation problems, or features
not working correctly when there is no physical hardware issue.

billing_purchases:
Charges, payments, purchases, refunds, subscriptions,
billing disputes, unexpected charges, App Store purchases,
or payment problems.

device_connectivity:
Problems connecting a device to Wi-Fi, Bluetooth,
cellular/mobile network, hotspot, AirDrop, or other
device/network connectivity.

general_inquiry:
General questions, requests for information, product questions,
availability questions, recommendations, greetings, or questions
that do not describe a specific technical problem.

other:
Messages that do not reasonably fit any of the six categories,
including ambiguous/noisy/irrelevant messages.
"""


# ============================================================
# FEW-SHOT CLASSIFICATION EXAMPLES
# ============================================================

CLASSIFICATION_EXAMPLES = """
EXAMPLES

Customer:
"My iPhone won't connect to Wi-Fi."

Correct:
{"intent":"device_connectivity","confidence_score":0.98,"should_escalate":false,"escalation_reason":""}

Customer:
"My iPhone screen is cracked after I dropped it."

Correct:
{"intent":"hardware_repair","confidence_score":0.99,"should_escalate":true,"escalation_reason":"Physical hardware damage"}

Customer:
"iOS keeps crashing after the latest update."

Correct:
{"intent":"software_troubleshooting","confidence_score":0.97,"should_escalate":false,"escalation_reason":""}

Customer:
"I forgot my Apple ID password and can't sign in."

Correct:
{"intent":"account_security","confidence_score":0.98,"should_escalate":false,"escalation_reason":""}

Customer:
"I was charged for an app I didn't buy. I want a refund."

Correct:
{"intent":"billing_purchases","confidence_score":0.99,"should_escalate":true,"escalation_reason":"Refund or disputed charge"}

Customer:
"Which iPhone should I buy?"

Correct:
{"intent":"general_inquiry","confidence_score":0.95,"should_escalate":false,"escalation_reason":""}

Customer:
"Your service is fucking useless. Nothing works."

Correct:
{"intent":"other","confidence_score":0.85,"should_escalate":true,"escalation_reason":"Highly frustrated or toxic customer"}
"""


class SupportAgent:

    def __init__(
        self,
        db_path=CHROMA_DB_PATH,
        collection_name="apple_support"
    ):

        # ====================================================
        # 1. LLM7 CLIENT
        # ====================================================

        api_key = (
            os.getenv("LLM7_API_TOKEN")
            or os.getenv("LLM7_API_KEY")
        )

        if not api_key:

            raise ValueError(
                "LLM7 API key was not found.\n\n"
                f"Expected .env file:\n{ENV_PATH}\n\n"
                "Add:\n"
                "LLM7_API_TOKEN=your_llm7_api_key_here"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=LLM7_BASE_URL,
            timeout=REQUEST_TIMEOUT
        )

        self.model = (
            os.getenv(
                "LLM7_MODEL",
                DEFAULT_LLM7_MODEL
            ).strip()
            or DEFAULT_LLM7_MODEL
        )

        print(f"🤖 LLM7 model: {self.model}")
        print(f"🌐 LLM7 API: {LLM7_BASE_URL}")

        # ====================================================
        # 2. CHROMADB
        # ====================================================

        db_path = Path(db_path)

        db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.chroma_client = chromadb.PersistentClient(
            path=str(db_path)
        )

        self.collection = (
            self.chroma_client.get_or_create_collection(
                name=collection_name
            )
        )

        print(
            f"📚 ChromaDB collection: "
            f"{collection_name}"
        )

        print(
            f"📊 Knowledge base records: "
            f"{self.collection.count()}"
        )


    # ========================================================
    # BUILD KNOWLEDGE BASE
    # ========================================================

    def build_knowledge_base(
        self,
        rag_csv_path=RAG_CSV_PATH
    ):

        if self.collection.count() > 0:

            print(
                f"✅ Knowledge base already loaded with "
                f"{self.collection.count()} records."
            )

            return

        print(
            "Building RAG Knowledge Base..."
        )

        rag_csv_path = Path(rag_csv_path)

        if not rag_csv_path.exists():

            raise FileNotFoundError(
                "\n❌ RAG CSV not found:\n"
                f"{rag_csv_path}"
            )

        df = pd.read_csv(rag_csv_path)

        required_columns = [
            "customer_text",
            "agent_text",
            "thread_id"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        print(
            f"📄 Loaded {len(df)} historical threads."
        )

        batch_size = 200

        for i in tqdm(
            range(
                0,
                len(df),
                batch_size
            ),
            desc="Loading knowledge base"
        ):

            batch = df.iloc[
                i:i + batch_size
            ]

            documents = (
                batch["customer_text"]
                .fillna("")
                .astype(str)
                .tolist()
            )

            metadatas = []

            for _, row in batch.iterrows():

                metadatas.append(
                    {
                        "agent_reply": str(
                            row["agent_text"]
                        )
                    }
                )

            ids = [
                str(tid)
                for tid in batch["thread_id"].tolist()
            ]

            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        print(
            f"✅ Knowledge base built: "
            f"{self.collection.count()} records."
        )


    # ========================================================
    # SIMPLE RULE-BASED SIGNALS
    # ========================================================

    @staticmethod
    def detect_rule_based_signals(text):

        text_lower = text.lower()

        signals = {
            "physical_damage": False,
            "refund_or_charge": False,
            "pii": False,
            "toxic": False,
        }

        # Physical damage
        physical_terms = [
            "cracked screen",
            "broken screen",
            "water damage",
            "water damaged",
            "dropped my phone",
            "broken button",
            "swollen battery",
            "bloated battery",
            "physical damage",
            "damaged screen",
        ]

        signals["physical_damage"] = any(
            term in text_lower
            for term in physical_terms
        )

        # Billing/refund
        billing_terms = [
            "refund",
            "charged",
            "charge",
            "charged me",
            "billing",
            "payment",
            "subscription",
            "money back",
            "didn't buy",
            "did not buy",
        ]

        signals["refund_or_charge"] = any(
            term in text_lower
            for term in billing_terms
        )

        # PII
        import re

        email_pattern = (
            r"\b[A-Za-z0-9._%+-]+@"
            r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )

        phone_pattern = (
            r"\b(?:\+?\d[\d\s().-]{7,}\d)\b"
        )

        signals["pii"] = bool(
            re.search(
                email_pattern,
                text
            )
            or
            re.search(
                phone_pattern,
                text
            )
        )

        # Toxicity
        toxic_terms = [
            "fuck",
            "fucking",
            "shit",
            "useless",
            "idiot",
            "stupid",
            "terrible",
            "worst",
            "hate",
        ]

        signals["toxic"] = any(
            term in text_lower
            for term in toxic_terms
        )

        return signals


    # ========================================================
    # CLASSIFY + TRIAGE
    # ========================================================

    def classify_and_triage(
        self,
        tweet_text
    ):

        signals = (
            self.detect_rule_based_signals(
                tweet_text
            )
        )

        prompt = f"""
You are an expert Apple Support intent classifier.

Your task is to classify ONE customer message.

{INTENT_DEFINITIONS}

{CLASSIFICATION_EXAMPLES}

ESCALATION POLICY

Set should_escalate=true when:

1. The customer requests a refund or disputes a charge.
2. There is physical hardware damage.
3. The customer provides personal information such as
   an email address or phone number.
4. The customer is highly toxic, abusive, or extremely frustrated.
5. The issue clearly requires human intervention.

Do NOT escalate merely because the problem is technical.

IMPORTANT CLASSIFICATION RULES

- Wi-Fi, Bluetooth, cellular, hotspot and AirDrop problems
  are device_connectivity.
- iOS bugs, crashes, update problems and software problems
  are software_troubleshooting.
- Physical damage belongs to hardware_repair.
- Refunds, charges and payments belong to billing_purchases.
- Apple ID/password/login problems belong to account_security.
- General product questions belong to general_inquiry.
- Use other only when none of the defined categories fit.
- Do not use general_inquiry as a default for a technical problem.
- Choose the most specific intent available.

Rule-based signals detected:

{json.dumps(signals, indent=2)}

Customer message:

"{tweet_text}"

Return ONLY valid JSON.

Required structure:

{{
    "intent": "one of the seven allowed intents",
    "confidence_score": 0.0,
    "should_escalate": false,
    "escalation_reason": ""
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
                        model=self.model,

                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a precise "
                                    "customer-support classifier. "
                                    "Return only JSON."
                                )
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],

                        temperature=0,

                        response_format={
                            "type": "json_object"
                        }
                    )
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
                        "Empty LLM response."
                    )

                if response_text.startswith("```"):

                    response_text = (
                        response_text
                        .replace("```json", "")
                        .replace("```JSON", "")
                        .replace("```", "")
                        .strip()
                    )

                data = json.loads(
                    response_text
                )

                if not isinstance(
                    data,
                    dict
                ):

                    raise ValueError(
                        "LLM response is not a JSON object."
                    )

                intent = str(
                    data.get(
                        "intent",
                        "other"
                    )
                ).strip()

                if intent not in VALID_INTENTS:

                    intent = "other"

                try:

                    confidence = float(
                        data.get(
                            "confidence_score",
                            0
                        )
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    confidence = 0.0

                confidence = max(
                    0.0,
                    min(
                        1.0,
                        confidence
                    )
                )

                should_escalate = bool(
                    data.get(
                        "should_escalate",
                        False
                    )
                )

                escalation_reason = str(
                    data.get(
                        "escalation_reason",
                        ""
                    )
                    or ""
                ).strip()

                # --------------------------------------------
                # Deterministic safety overrides
                # --------------------------------------------

                if signals["physical_damage"]:

                    intent = "hardware_repair"
                    should_escalate = True
                    escalation_reason = (
                        "Physical hardware damage"
                    )

                elif signals["refund_or_charge"]:

                    # Only force billing if the message is
                    # actually about a payment/charge/refund.
                    if intent in {
                        "billing_purchases",
                        "general_inquiry",
                        "other"
                    }:

                        intent = "billing_purchases"

                    should_escalate = True
                    escalation_reason = (
                        "Refund or disputed charge/payment issue"
                    )

                elif signals["pii"]:

                    should_escalate = True
                    escalation_reason = (
                        "Customer message contains possible PII"
                    )

                elif signals["toxic"]:

                    should_escalate = True
                    escalation_reason = (
                        "Highly frustrated or toxic customer"
                    )

                data = {
                    "intent": intent,
                    "confidence_score": confidence,
                    "should_escalate": should_escalate,
                    "escalation_reason": escalation_reason
                }

                return data

            except Exception as e:

                if attempt >= MAX_RETRIES:

                    raise RuntimeError(
                        "LLM7 classification failed after "
                        f"{MAX_RETRIES} attempts.\n"
                        f"Model: {self.model}\n"
                        f"Error: {e}"
                    ) from e

                wait_time = attempt * 2

                print(
                    f"⚠️ Classification retry "
                    f"{attempt}/{MAX_RETRIES} "
                    f"in {wait_time}s..."
                )

                time.sleep(
                    wait_time
                )


    # ========================================================
    # RAG RETRIEVAL
    # ========================================================

    def retrieve_similar_issues(
        self,
        tweet_text,
        n_results=5
    ):

        if self.collection.count() == 0:

            return []

        results = self.collection.query(
            query_texts=[
                tweet_text
            ],
            n_results=min(
                n_results,
                self.collection.count()
            )
        )

        documents = (
            results.get("documents")
            or [[]]
        )

        metadatas = (
            results.get("metadatas")
            or [[]]
        )

        retrieved = []

        if documents and documents[0]:

            for i, document in enumerate(
                documents[0]
            ):

                metadata = {}

                if (
                    metadatas
                    and
                    metadatas[0]
                    and
                    i < len(metadatas[0])
                ):

                    metadata = (
                        metadatas[0][i]
                        or {}
                    )

                historical_reply = str(
                    metadata.get(
                        "agent_reply",
                        ""
                    )
                )

                retrieved.append(
                    {
                        "customer_issue": str(
                            document
                        ),
                        "agent_reply": historical_reply
                    }
                )

        return retrieved


    # ========================================================
    # DRAFT REPLY
    # ========================================================

    def draft_reply(
        self,
        tweet_text,
        retrieved_examples
    ):

        if not retrieved_examples:

            context_str = (
                "No historical examples were retrieved."
            )

        else:

            context_parts = []

            for i, example in enumerate(
                retrieved_examples,
                start=1
            ):

                context_parts.append(
                    f"""
Historical Example {i}

Customer:
{example["customer_issue"]}

Historical Agent Reply:
{example["agent_reply"]}
"""
                )

            context_str = "\n".join(
                context_parts
            )

        prompt = f"""
You are an Apple Support customer-service agent.

Write a concise, helpful and empathetic reply.

Customer message:
"{tweet_text}"

Historical support examples:
{context_str}

RULES:

- Use the historical examples as grounding.
- Do not invent Apple policies.
- Do not invent URLs.
- Do not claim that you performed an action.
- Do not promise refunds, replacements, repairs or outcomes.
- Ask for the minimum useful information needed.
- Give a practical next step when the examples support it.
- Keep the reply under 280 characters.
- Return ONLY the customer-facing reply.
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
                        model=self.model,

                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an empathetic "
                                    "Apple Support agent. "
                                    "Be concise and grounded."
                                )
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],

                        temperature=0.2
                    )
                )

                if not response.choices:

                    raise ValueError(
                        "LLM7 returned no reply choices."
                    )

                reply = (
                    response
                    .choices[0]
                    .message
                    .content
                    or ""
                ).strip()

                if not reply:

                    raise ValueError(
                        "LLM7 returned an empty reply."
                    )

                if (
                    len(reply) >= 2
                    and reply.startswith('"')
                    and reply.endswith('"')
                ):

                    reply = (
                        reply[1:-1]
                        .strip()
                    )

                if len(reply) > 280:

                    reply = (
                        reply[:277]
                        .rstrip()
                        + "..."
                    )

                return reply

            except Exception as e:

                if attempt >= MAX_RETRIES:

                    raise RuntimeError(
                        "LLM7 reply generation failed "
                        f"after {MAX_RETRIES} attempts.\n"
                        f"Model: {self.model}\n"
                        f"Error: {e}"
                    ) from e

                wait_time = attempt * 2

                print(
                    f"⚠️ Reply retry "
                    f"{attempt}/{MAX_RETRIES} "
                    f"in {wait_time}s..."
                )

                time.sleep(
                    wait_time
                )


    # ========================================================
    # PROCESS TWEET
    # ========================================================

    def process_tweet(
        self,
        tweet_text
    ):

        if tweet_text is None:

            raise ValueError(
                "tweet_text cannot be None."
            )

        tweet_text = str(
            tweet_text
        ).strip()

        if not tweet_text:

            raise ValueError(
                "tweet_text cannot be empty."
            )

        # ----------------------------------------------------
        # 1. Classification
        # ----------------------------------------------------

        analysis = (
            self.classify_and_triage(
                tweet_text
            )
        )

        confidence = float(
            analysis.get(
                "confidence_score",
                0
            )
        )

        should_escalate = bool(
            analysis.get(
                "should_escalate",
                False
            )
        )

        # ----------------------------------------------------
        # 2. Low-confidence cases go to human
        # ----------------------------------------------------

        if (
            should_escalate
            or
            confidence < 0.60
        ):

            return {
                "status": "escalated",
                "analysis": analysis,
                "reply": None
            }

        # ----------------------------------------------------
        # 3. Retrieve historical examples
        # ----------------------------------------------------

        retrieved_examples = (
            self.retrieve_similar_issues(
                tweet_text,
                n_results=5
            )
        )

        # ----------------------------------------------------
        # 4. Generate grounded response
        # ----------------------------------------------------

        drafted_reply = (
            self.draft_reply(
                tweet_text,
                retrieved_examples
            )
        )

        return {
            "status": "auto_handled",
            "analysis": analysis,
            "reply": drafted_reply,
            "retrieved_examples": len(
                retrieved_examples
            )
        }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 60
    )

    print(
        "🧪 Testing SupportAgent with LLM7"
    )

    print(
        "=" * 60
    )

    try:

        agent = SupportAgent()

        test_messages = [
            (
                "My iPhone won't connect to Wi-Fi. "
                "I've tried restarting it but nothing works."
            ),

            (
                "My iPhone screen is cracked after "
                "I dropped it."
            ),

            (
                "iOS keeps crashing after the latest update."
            ),

            (
                "I was charged for an app I didn't buy. "
                "I want a refund."
            ),
        ]

        for test_message in test_messages:

            print(
                "\n" + "-" * 60
            )

            print(
                f"👤 Customer:\n{test_message}"
            )

            result = agent.process_tweet(
                test_message
            )

            print(
                "\n🤖 Result:"
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False
                )
            )

    except Exception as e:

        print(
            "\n❌ Error:"
        )

        print(e)
