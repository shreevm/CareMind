class ClarificationAgent:
    """Requests more context when the supervisor cannot pick a useful evidence source."""

    def answer(self) -> str:
        return "Which uploaded report or medical topic should I use as the evidence source?"
