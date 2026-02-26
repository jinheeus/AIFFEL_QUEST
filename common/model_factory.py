from langchain_naver import ChatClovaX

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    pass
from langchain_openai import ChatOpenAI
from common.config import Config
from common.logger_config import setup_logger

logger = setup_logger("ModelFactory")


class ModelFactory:
    """
    Centralized Model Factory for Agentic RAG.
    - RAG Agents (SOP, Adversarial) -> HyperCLOVA X
    - Evaluation -> Configurable (Gemini vs OpenAI)
    """

    @staticmethod
    def get_rag_model(level: str = "light", temperature: float = 0.1):
        """
        Returns HyperCLOVA X model for RAG tasks.
        :param level: 'light' (HCX-DASH) or 'heavy' (HCX-003)
        """
        if level == "reasoning":
            model_name = Config.HCX_MODEL_REASONING
            logger.info(f"RAG Agent using {model_name} (Level: {level})")
            return ChatClovaX(
                model=model_name,
                max_tokens=4096,  # Higher limit for deep thinking output
                temperature=0.2,  # Slight creativity for complex analysis
            )
        elif level == "heavy":
            model_name = Config.HCX_MODEL_HEAVY  # Defaults to STANDARD (003)
            logger.info(f"RAG Agent using {model_name} (Level: {level})")
            return ChatClovaX(
                model=model_name,
                max_tokens=2048,
                # temperature parameter omitted for safety with HCX-003
            )
        else:
            model_name = Config.HCX_MODEL_LIGHT
            logger.info(f"RAG Agent using {model_name} (Level: {level})")
            return ChatClovaX(
                model=model_name,
                temperature=temperature,
                max_tokens=1024,
            )

    @staticmethod
    def get_eval_model(level: str = "light", temperature: float = 0.0):
        """
        Returns Evaluation model based on Config.EVAL_PROVIDER.
        :param level: 'light' (Flash/Mini) or 'heavy' (Pro/GPT-4o)
        """
        provider = Config.EVAL_PROVIDER.lower()

        if provider == "gemini":
            model_name = (
                Config.GEMINI_MODEL_HEAVY
                if level == "heavy"
                else Config.GEMINI_MODEL_LIGHT
            )
            logger.info(f"Evaluation using Gemini: {model_name}")
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

        elif provider == "openai":
            model_name = (
                Config.OPENAI_MODEL_HEAVY
                if level == "heavy"
                else Config.OPENAI_MODEL_LIGHT
            )
            logger.info(f"Evaluation using OpenAI: {model_name}")
            return ChatOpenAI(model=model_name, temperature=temperature)

        else:
            raise ValueError(f"Unknown EVAL_PROVIDER: {provider}")
