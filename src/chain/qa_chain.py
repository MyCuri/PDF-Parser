from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

import config


class QAChain:
    def __init__(self, retriever, system_prompt: str = "You are a helpful assistant."):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o",
            api_key=config.OPENAI_API_KEY,
        )
        self.retriever = retriever
        self.system_prompt = system_prompt
        self._setup_chain()

    def format_docs(self, docs):
        return "\n\n".join([d.page_content for d in docs])

    def _prompt_func(self, data):
        """Build the prompt for the LLM based on the context and question."""

        prompt = f"""Answer the question based only on the context.
        Question: {data["question"]}
        
        Context:
        {data["context"]}
        """

        return [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]

    def _setup_chain(self):
        """Set up the QA chain with all necessary components."""
        self.chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self._prompt_func)
            | self.llm
            | StrOutputParser()
        )

    def run(self, user_query):
        """Run the QA chain with a query."""
        return self.chain.invoke(user_query)
