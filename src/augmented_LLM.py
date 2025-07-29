from openai import OpenAI
from src.retrieve import retrieve_top_k


def generate(question):

    client = OpenAI(
                base_url="https://router.huggingface.co/v1",

            )

    # with open("src/temp_question.txt") as f:
    #     question = f.read().strip()

    context = "\n".join(retrieve_top_k(question, 5))
    instructions = """
    1.  Read the provided "Context" carefully.
    2.  Answer the "Question" based *strictly* on the information
        available in the Context.
    3.  If the Context contains enough information to fully answer the
        question, provide *only* the "Final Answer" directly, without any
        preamble, numbering, or step-by-step breakdown.
    4.  If the Context *does not* contain enough information to
        fully answer the question, or if the question cannot be answered from
        the Context, state clearly: "The provided context does not contain
        enough information to answer this question fully." Do not attempt to
        guess or use outside knowledge unless explicitly instructed otherwise
        for a specific type of query.
        """

    prompt = f"Instructions:\n{instructions}\nContext:\n{context}\n\nQuestion\
        : {question}\nAnswer:"

    # Make chat completion request
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Output response
    print("\n🔍 Answer:\n", response.choices[0].message.content)
    return response.choices[0].message.content
