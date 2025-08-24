from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional


load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "must write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, 'must write down a brief summary of the review']
    sentiment: Annotated[str, 'must return sentiment of the review, either Positive, Negative or Neutral']
    pros: Annotated[Optional[list[str]], 'write down all the pros inside a list' ]
    cons: Annotated[Optional[list[str]], 'write down all th cons inside a list' ]



structured_model=model.with_structured_output(Review, strict=True)

result=structured_model.invoke("""
Google’s Pixel phones have never been the most powerful handsets, with their Tensor chipsets falling behind rivals in benchmarks. But surprisingly, the Google Pixel 10 series might be even more compromised than the Pixel 9 series, at least when it comes to the GPU (Graphics Processing Unit).

In a post on X, @lafaiel (via Phone Arena) shared a screenshot of a Geekbench listing for the Google Pixel 10 Pro, in which the Pixel 10 Pro achieved a GPU score of just 3,707. Higher is better here, and for comparison, the Pixel 9 Pro’s score is 9,023, while rivals like the Samsung Galaxy S25 Plus and iPhone 16 Pro achieve scores of 26,333 and 33,374 respectively.

So based on this result the Pixel 10 Pro is way behind, though the fact that it scored even less than its predecessor is especially worrying.
                               Performance upgrades in other areas

Now, GPU performance is only one part of the power picture, and the Pixel 10 Pro should outperform the Pixel 9 Pro on Geekbench overall, with the same source recording a single core score of 2,329 and a multi-core result of 6,358 for the similarly spec'd Google Pixel 10 Pro XL, compared to 1,948 and 4,530 for single and multi-core repectively on the Pixel 9 Pro XL.

But such a low GPU score means the Google Pixel 10 Pro might still struggle with demanding games.

Google claims the Pixel 10 series will feel faster in everday use, and perform better for AI tasks, and that’s probably true, so if you don’t care about games then this shouldn’t affect you much. But if you’re a big mobile gamer then you might want to think twice about buying the Pixel 10 Pro – or any Pixel 10 model, as they all use the same Tensor G5 chipset, so will probably all have similar GPU performance.

That said, while this looks like a legitimate Geekbench screenshot, it’s still just one result, so it’s possible this will turn out to be an outlier. It’s also feasible that Google might be able to improve the Pixel 10 Pro’s GPU performance through a software update.
                                
                              """
)
print(result)