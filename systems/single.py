"""The single-call variant: one model call per request, no steps.

The model does only what needs language: it reads the request into structured requirements, and it reads each
game's description (features such as fighting or horror, wording about how easy the rules are, and numbers the
card doesn't list). Plain Python then checks every requirement against the cards, following RUBRIC.md, and picks.
Still one model call per request.
"""
import re
from typing import Literal, Optional, Union

from pydantic import BaseModel

from p1 import Answer, Request, call, parse_json
from p1.render import render_request
from p1.types import Candidate

VARIANT = "single"

FEATURES = ("fighting", "violence", "horror", "timer", "elimination")

# Your instructions: what the model should do with the request and the five games.
INSTRUCTIONS = """You help recommend board games. You get a person's request and five candidate games (A to E), each \
with a fact card and a description. You do NOT pick a game. You do two jobs, and code does the rest.

JOB 1: Read the request into requirements. Only write down what the request actually asks for. Never invent a \
requirement: if the request doesn't mention play time, max_minutes is null, and so on. For every requirement except \
players, copy into its *_quote field the exact words of the request that state it. If you can't quote words that \
state it, the requirement is null. Code checks the quotes against the request.
Wishes are not requirements: ignore everything after "would be nice", "would be amazing", "we'd love", "it would be \
great", and "something like X".
- players: count everyone who plays, including the writer when they say "I", "we", "us" or "me". "Me and my brother" \
= 2. "My wife and I" = 2. "My friend and I" = 2. "Me and my three friends" = 4. "Six of us" = 6. "Me, my husband and \
our three kids" = 5. "Game night with six friends" = 7.
- max_minutes: "an hour", "an hour tops", "about an hour" = 60. "Half an hour" = 30. "No more than 90 minutes" = 90. \
"Two hours at most" = 120.
- complexity: "easy rules", "simple", "beginners", "never play board games", "casual players" = "light". "Deep \
strategy", "meaty", "nothing light" = "heavy". "Not too simple, not too heavy" = "medium".
- mode: "together against the game", "team up against the game", "as one team", "work together against the game" \
= "cooperative". "Head to head", "against each other" = "competitive". Otherwise null. "We play together", "play \
together a lot" or "get together" only mean they play in the same group: that is NOT cooperative, so mode is null.
- child_age: "our 8-year-old" = 8. Otherwise null.
- avoid: "no fighting"/"no combat" = "fighting". "Nothing violent" = "violence". "Nothing scary"/"nothing creepy" = \
"horror". "No timers"/"no racing against the clock" = "timer". "Nobody knocked out"/"nobody sitting out" = \
"elimination". Only list what the request says, each with its quote.

JOB 2: For each game, read its DESCRIPTION (the card is checked by code).
- features: which of the features in your avoid list the description shows actually happening when people play, \
each with the exact words of the description that show it. Only check the features in your avoid list; if avoid \
is empty, write []. Use these definitions:
  fighting: the players' characters or creatures attack, battle or duel other characters or creatures. NOT \
fighting: armies, fleets or units attacking on a map, as in wargames (that is violence); raiding or conquering to score; metaphors ("disease-fighting", \
"fight over resources", "battle to spread plagues"); backstory; contests only called duels or battles; a penalty \
called an attack (a pirate or thieves stealing cards); abstract captures like chess; a genre label alone.
  violence: the game is about war, battles between armies, raiding, pillaging, conquest or killing, or it has \
fighting; this includes players defending against or confronting raiders, invaders or enemies as part of play. \
NOT violence: a penalty called an attack, metaphors, a grim backstory.
  horror: the game is meant to frighten: vampires, zombies, werewolves or Cthulhu monsters as the threat, \
threatening ghosts or hauntings, being hunted, frightening gore. NOT horror: generic fantasy monsters and dungeons, \
harmless or helpful ghosts, undead or demons as enemies in heroic or comic fantasy, a horror word only in a \
licensed title, realistic threats, cartoon spookiness in family or children's games, one horror character among \
many, giant monsters smashing a city, murder mysteries.
  timer: racing a clock or sand timer, everyone playing at once as fast as they can, grabbing or running at the \
same moment, timed turns. NOT a timer: a countdown track that ends the game, choosing actions at the same time \
without racing, "fast-paced", racing only as a theme, an optional timer.
  elimination: a player can be knocked out and must sit and watch, for the rest of the game or of a round (also \
when the rules make it unavoidable: "last person alive wins", or "win all the cards" with 3 or more players). NOT \
elimination: losing points or pieces while still playing, a cooperative team losing together, a 2-player game that \
ends when one player is knocked out, a knocked-out player who keeps playing in another role.
  A feature counts even if it is only in an optional variant or one scenario. If the description contradicts \
itself, the more detailed rule wins.
- rules: what the description SAYS about how hard the game's rules are. Do NOT judge from the weight, the game \
types or the theme: code already uses the card. At the end, under "Sentences about difficulty", code lists for \
each game the description sentences that might be about this; read them carefully. Write "easy" if one says the \
rules or the game are easy, simple, easy to learn, streamlined, or lighter than similar games. Write "complex" if \
one says the rules or the game are complex, very complex, for experienced or expert players, or heavier than \
similar games. Then copy the exact words into rules_quote. Write "none" (rules_quote null) when no sentence is \
listed, or a sentence says both ("easy to learn, hard to master"), or it is about how hard the game is to WIN \
("difficult to survive"), about one part only or an optional variant, or about the audience ("family game").
- players, minutes, age: ONLY for card fields listed under "Card fields not listed" at the end. Give the number \
the description states for the base game (players as [min, max], minutes as the longest time, age as the minimum \
age), or null if it doesn't state one. Time to learn the rules is not play time. Player counts that need a \
separately sold expansion don't count. For every other game, write null."""

# The answer format. Keep it, unless you also change how answer() reads the reply.
FORMAT = """Answer with JSON only, in this form (the values are only an example):
{
  "requirements": {
    "players": 4,
    "max_minutes": 60, "time_quote": "an hour tops",
    "complexity": "light", "complexity_quote": "we need easy rules",
    "mode": null, "mode_quote": null,
    "child_age": null, "age_quote": null,
    "avoid": [{"feature": "horror", "quote": "nothing scary"}]
  },
  "games": {
    "A": {"features": [], "rules": "none", "rules_quote": null, "players": null, "minutes": null, "age": null},
    "B": {"features": [{"feature": "horror", "quote": "zombies hunt the players"}], "rules": "easy", \
"rules_quote": "the rules are simple", "players": null, "minutes": null, "age": null},
    "C": {"features": [], "rules": "none", "rules_quote": null, "players": [2, 6], "minutes": null, "age": null},
    "D": {"features": [], "rules": "none", "rules_quote": null, "players": null, "minutes": null, "age": null},
    "E": {"features": [], "rules": "none", "rules_quote": null, "players": null, "minutes": null, "age": null}
  }
}
Use null (no quotes) for anything missing."""


class Quoted(BaseModel):
    feature: Optional[str] = None
    quote: Optional[str] = None


class Requirements(BaseModel):
    players: Optional[int] = None
    max_minutes: Optional[int] = None
    time_quote: Optional[str] = None
    complexity: Optional[str] = None
    complexity_quote: Optional[str] = None
    mode: Optional[str] = None
    mode_quote: Optional[str] = None
    child_age: Optional[int] = None
    age_quote: Optional[str] = None
    avoid: list[Quoted] = []


class GameReading(BaseModel):
    features: list[Quoted] = []
    rules: Optional[str] = "none"
    rules_quote: Optional[str] = None
    players: Optional[Union[list[int], int]] = None
    minutes: Optional[Union[list[int], int]] = None
    age: Optional[int] = None


class Reply(BaseModel):
    requirements: Requirements = Requirements()
    games: dict[str, GameReading] = {}


Verdict = Literal["fits", "breaks", "unknown"]


def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _missing_fields(c: Candidate) -> list[str]:
    card = c.card
    return [name for name, value in (("players", card.players), ("minutes", card.minutes), ("age", card.age))
            if value is None]


def _missing_note(request: Request) -> str:
    lines = [f"{c.id}: {', '.join(_missing_fields(c))}" for c in request.candidates if _missing_fields(c)]
    return "Card fields not listed:\n" + ("\n".join(lines) if lines else "none")


DIFFICULTY = re.compile(r"\b(simple|simpler|easy|easier|easily|complex|complexity|complicated|expert|experienced|"
                        r"beginners?|novices?|newcomers?|light|lighter|heavy|heavier|streamlined|straightforward|"
                        r"accessible|learn|depth|casual|gamers?)\b", re.I)


def difficulty_sentences(c: Candidate) -> list[str]:
    """The description's sentences (or lines) that use a word about difficulty."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", c.description)
    return [p.strip()[:300] for p in parts if DIFFICULTY.search(p)][:4]


def _difficulty_note(request: Request) -> str:
    lines = []
    for c in request.candidates:
        found = difficulty_sentences(c)
        lines.append(f"{c.id}: " + (" | ".join(f'"{x}"' for x in found) if found else "none listed"))
    return "Sentences about difficulty:\n" + "\n".join(lines)


NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
_N = r"(\d{1,2}|" + "|".join(NUMBERS) + r")"
PLAYER_RANGE = re.compile(_N + r"\s*(?:-|\u2013|to)\s*" + _N + r"\s+players", re.I)


def players_in_description(c: Candidate) -> Optional[tuple[int, int]]:
    """A player range the description states ("2-4 players", "two to six players"), outside any sentence about an
    expansion. A fallback for when the card doesn't list players and the model found none."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", c.description):
        if "expansion" in sentence.lower():
            continue
        m = PLAYER_RANGE.search(sentence)
        if m:
            lo, hi = (int(x) if x.isdigit() else NUMBERS[x.lower()] for x in m.groups())
            if 1 <= lo <= hi <= 20:
                return lo, hi
    return None


def _as_range(value) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, int):
        return value, value
    if len(value) == 0:
        return None
    return min(value), max(value)


# Rubric section 3, complexity: each question is "yes", "no" or "unknown".
def _light(weight: Optional[float], types: set[str], rules: str) -> str:
    if weight is not None and weight <= 1.3:
        return "yes"
    if weight is not None and weight >= 4.0:
        return "no"
    yes = rules == "easy" or bool(types & {"children's", "party"}) or (
        "family" in types and weight is not None and weight < 2.0)
    no = rules == "complex" or (
        bool(types) and types <= {"strategy", "thematic", "war", "abstract"} and weight is not None and weight >= 2.5)
    return "unknown" if yes == no else ("yes" if yes else "no")


def _heavy(weight: Optional[float], types: set[str], rules: str) -> str:
    if weight is not None and weight >= 4.0:
        return "yes"
    if weight is not None and weight <= 1.3:
        return "no"
    yes = rules == "complex" or (
        bool(types & {"strategy", "war"}) and "family" not in types and weight is not None and weight >= 3.5)
    no = rules == "easy" or bool(types & {"children's", "party", "family"})
    return "unknown" if yes == no else ("yes" if yes else "no")


def _medium(weight: Optional[float], light: str, heavy: str) -> str:
    if light == "yes" or heavy == "yes":
        return "no"
    if weight is not None and 2.0 <= weight <= 3.5:
        return "yes"
    return "unknown"


def complexity_class(c: Candidate, rules: str) -> dict[str, str]:
    weight = c.card.complexity.weight if c.card.complexity else None
    types = {_norm(t) for t in c.card.types}
    light = _light(weight, types, rules)
    heavy = _heavy(weight, types, rules)
    return {"light": light, "heavy": heavy, "medium": _medium(weight, light, heavy)}


def has_tag(c: Candidate, feature: str) -> bool:
    cats = {_norm(x) for x in c.card.categories}
    mechs = {_norm(x) for x in c.card.mechanics}
    if feature == "fighting":
        return "fighting" in cats
    if feature == "violence":
        return bool(cats & {"wargame", "fighting"})
    if feature == "horror":
        return bool(cats & {"horror", "zombies"})
    if feature == "timer":
        return "real-time" in cats or "real-time" in mechs
    if feature == "elimination":
        return "player elimination" in mechs
    return False


# The model must back what it reads with a quote. Code keeps only what the quotes support.
COOP_WORDS = ("against the game", "team up", "one team", "cooperative", "co-op", "work together", "working together")
COMPETITIVE_WORDS = ("against each other", "head to head", "against one another", "competitive", "compete",
                     "free for all", "out for themselves", "beat the other")
EASY_WORDS = ("easy", "easier", "simple", "light", "straightforward", "accessible", "streamlined", "learn")
COMPLEX_WORDS = ("complex", "complicated", "experienced", "expert", "heav", "advanced", "veteran")
TIME_WORDS = ("minute", "hour", "min", "half", "quick", "short")
WISH = re.compile(r"would be nice|would be amazing|would be great|we'd love|would love|not a must|"
                  r"doesn't have to|does not have to|bonus points", re.I)
STOPWORDS = {"and", "or", "the", "a", "an", "but"}


def _flat(text: Optional[str]) -> str:
    """Lower case, with every run of non-letters and non-digits turned into one space."""
    return " ".join(re.sub(r"[^a-z0-9']+", " ", (text or "").lower().replace("\u2019", "'")).split())


def _content(flat: str) -> str:
    return " ".join(w for w in flat.split() if w not in STOPWORDS)


def quoted_in(quote: Optional[str], text: str) -> bool:
    """Is the quote (pieces split by "...") really in the text? Small words such as "and" may differ, as long as a
    piece keeps at least two other words."""
    pieces = [_flat(p) for p in re.split(r"\.\.\.|\u2026", quote or "")]
    pieces = [p for p in pieces if p]
    flat = f" {_flat(text)} "
    loose = f" {_content(_flat(text))} "

    def found(piece: str) -> bool:
        core = _content(piece)
        return f" {piece} " in flat or (len(core.split()) >= 2 and f" {core} " in loose)
    return bool(pieces) and all(found(p) for p in pieces)


def in_wish(quote: Optional[str], text: str) -> bool:
    """Is the quote inside a sentence that states a wish ("would be nice", "not a must")?"""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if quoted_in(quote, sentence) and WISH.search(sentence):
            return True
    return False


def clean_requirements(req: Requirements, request_text: str) -> Requirements:
    """Drop each requirement whose quote isn't in the request, and a mode whose quote doesn't say it."""
    def ok(quote):
        return quoted_in(quote, request_text)
    mode = _norm(req.mode)
    words = {"cooperative": COOP_WORDS, "competitive": COMPETITIVE_WORDS}.get(mode, ())
    mode_ok = (ok(req.mode_quote) and any(w in _flat(req.mode_quote) for w in words)
               and not in_wish(req.mode_quote, request_text))
    time_ok = ok(req.time_quote) and any(w in _flat(req.time_quote) for w in TIME_WORDS)
    return Requirements(
        players=req.players,
        max_minutes=req.max_minutes if time_ok else None,
        complexity=req.complexity if ok(req.complexity_quote) else None,
        mode=mode if mode_ok else None,
        child_age=req.child_age if ok(req.age_quote) else None,
        avoid=[q for q in req.avoid if _norm(q.feature) in FEATURES and ok(q.quote)],
    )


def rules_wording(c: Candidate, reading: GameReading) -> str:
    """The model's "easy"/"complex", kept only if its quote is in the description and says so."""
    rules = _norm(reading.rules)
    words = {"easy": EASY_WORDS, "complex": COMPLEX_WORDS}.get(rules)
    if words and quoted_in(reading.rules_quote, c.description) and any(w in _flat(reading.rules_quote) for w in words):
        return rules
    return "none"


def check_game(c: Candidate, req: Requirements, reading: GameReading) -> dict[str, Verdict]:
    """Check each requirement of the request against one game."""
    checks: dict[str, Verdict] = {}
    card = c.card
    if req.players:
        rng = _as_range(card.players) or _as_range(reading.players) or players_in_description(c)
        checks["players"] = "unknown" if rng is None else ("fits" if rng[0] <= req.players <= rng[1] else "breaks")
    if req.max_minutes:
        rng = _as_range(card.minutes) or _as_range(reading.minutes)
        checks["play time"] = "unknown" if rng is None else ("fits" if rng[1] <= req.max_minutes else "breaks")
    if req.child_age:
        age = card.age if card.age is not None else reading.age
        checks["age"] = "unknown" if age is None else ("fits" if age <= req.child_age else "breaks")
    level = _norm(req.complexity)
    if level in ("light", "medium", "heavy"):
        answer = complexity_class(c, rules_wording(c, reading))[level]
        checks["complexity"] = {"yes": "fits", "no": "breaks"}.get(answer, "unknown")
    mode = _norm(req.mode)
    if mode in ("cooperative", "competitive"):
        coop = "cooperative game" in {_norm(m) for m in card.mechanics}
        checks[mode] = "fits" if coop == (mode == "cooperative") else "breaks"
    seen = {_norm(q.feature) for q in reading.features if quoted_in(q.quote, c.description)}
    for feature in {_norm(q.feature) for q in req.avoid} & set(FEATURES):
        # Nothing violent also rules out fighting.
        related = {feature, "fighting"} if feature == "violence" else {feature}
        has = any(has_tag(c, f) or f in seen for f in related)
        checks[f"no {feature}"] = "breaks" if has else "fits"
    return checks


def _explain(c: Candidate, req: Requirements, checks: dict[str, Verdict]) -> str:
    facts = []
    card = c.card
    if "players" in checks and card.players:
        facts.append(f"it plays {card.players[0]} to {card.players[-1]} and you are {req.players}")
    if "play time" in checks and card.minutes:
        facts.append(f"it takes at most {card.minutes[-1]} minutes")
    if "complexity" in checks:
        facts.append(f"its rules fit your wish for a {req.complexity} game")
    for name in checks:
        if name.startswith("no ") or name in ("cooperative", "competitive", "age"):
            facts.append(f"it fits '{name}'")
    return f"{c.name} meets every requirement: " + "; ".join(facts) + "." if facts else f"{c.name} fits your request."


def decide(request: Request, reply: Reply) -> Answer:
    req = clean_requirements(reply.requirements, request.request)
    reasons = []
    for c in request.candidates:
        reading = reply.games.get(c.id) or GameReading()
        checks = check_game(c, req, reading)
        if all(v == "fits" for v in checks.values()):
            return Answer(pick=c.id, explanation=_explain(c, req, checks))
        bad = [f"{k} ({v})" for k, v in checks.items() if v != "fits"]
        reasons.append(f"{c.name}: {', '.join(bad)}")
    return Answer(pick=None, explanation="None of the games surely meets every requirement. " + "; ".join(reasons) + ".")


def answer(request: Request) -> Answer:
    if not INSTRUCTIONS.strip():
        raise NotImplementedError("Write your instructions in INSTRUCTIONS in systems/single.py first.")
    prompt = (f"{INSTRUCTIONS}\n\n{FORMAT}\n\n{render_request(request)}\n\n{_missing_note(request)}\n\n"
              f"{_difficulty_note(request)}")
    # One call only, so we can't ask again: a failed call, a reply we can't read, or a bug in the checks becomes a
    # decline, so every request still gets an answer.
    try:
        text = call(prompt, max_tokens=1500)
        reply = parse_json(text, Reply)
        return decide(request, reply)
    except Exception:
        return Answer(pick=None, explanation="Sorry, I couldn't find a game that surely fits your request.")
