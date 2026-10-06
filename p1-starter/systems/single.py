"""The single-call variant: one model call per request, no steps.

Write your instructions for the model in INSTRUCTIONS. Keep it to one call: the starter code refuses a second
call and the request fails.
"""
from p1 import Answer, Request, call_json
from p1.render import render_request

VARIANT = "single"

# Your instructions: what the model should do with the request and the five games.
INSTRUCTIONS = """You recommend board games. Below is a person's request and five candidate games (A to E). Each game has a \
fact card and BoardGameGeek's description. Pick ONE game that surely meets every requirement in the request, or \
decline (pick null) if no game does. Use only the card and the description: ignore anything you know about the \
game from elsewhere.

STEP 1: List the requirements.
- Only hard requirements count. Wishes ("would be nice", "would be amazing", "we'd love", "something like X") \
are NOT requirements: ignore them.
- Players: count everyone who plays, including the writer when they say "I", "we", "us" or "me". \
"Me and my brother" = 2. "Me, my husband and our three kids" = 5. "Game night with six friends" = 7.
- Play time: turn phrases into a maximum in minutes. "An hour", "an hour tops", "about an hour" = at most 60. \
"Half an hour" = at most 30.
- Complexity: "easy rules", "simple", "beginners", "never play board games", "casual players" = LIGHT. \
"Deep strategy", "meaty", "nothing light" = HEAVY. "Not too simple, not too heavy" = MEDIUM.
- Cooperative: "together against the game", "team up against the game", "as one team". \
Competitive: "head to head", "against each other". If the request says nothing, there is no requirement.
- Age: "our 8-year-old" means the minimum age must be 8 or lower.
- Things to avoid: "no fighting"/"no combat" = no FIGHTING. "Nothing violent" = no VIOLENCE (which also rules out \
fighting). "Nothing scary"/"nothing creepy" = no HORROR. "No timers"/"no racing against the clock" = no TIMER. \
"Nobody knocked out"/"nobody sitting out" = no PLAYER ELIMINATION.

STEP 2: Check every requirement for every game. Each check is FITS, BREAKS or UNKNOWN.
- Players: fits if the number is within the card's range. If the card says "not listed", a number stated in the \
description counts. Only the base game counts: player counts that need a separately sold expansion don't.
- Play time: the game's LONGEST listed time must be within the limit ("30 to 60 minutes" does NOT fit "at most \
45"). If the card says "not listed", a play time stated in the description counts. Time to learn the rules is not \
play time.
- Age: fits if the card's minimum age is at or below the child's age. If not listed, an age in the description \
counts ("ages 8 and up"). "Family game" says nothing about age.
- Cooperative: the game is cooperative if and only if its mechanics include "Cooperative Game". Otherwise it is \
competitive (teams playing against each other are competitive).
- Complexity, using the weight W (1 to 5), the game types, and description wording about how hard the RULES are:
  LIGHT is YES if W <= 1.3, or the description says the rules are easy/simple, or it is a children's or party \
game, or it is a family game with W < 2.0.
  LIGHT is NO if W >= 4.0, or the description says the rules are complex/for experienced players, or its only \
types are strategy/thematic/war/abstract and W >= 2.5.
  HEAVY is YES if W >= 4.0, or the description says the rules are complex/for experienced players, or it is a \
strategy or war game (not also family) with W >= 3.5.
  HEAVY is NO if W <= 1.3, or the description says the rules are easy/simple, or it is a children's, party or \
family game.
  The extremes (W <= 1.3, W >= 4.0) beat every other clue. If clues point both ways, or none applies: UNKNOWN.
  MEDIUM is NO if the game is light or heavy; YES if it is neither and 2.0 <= W <= 3.5; otherwise UNKNOWN.
  Description wording counts only if it is about how hard the rules are ("lighter than many similar games"). \
"Easy to learn, hard to master" doesn't decide. Wording about how hard it is to win, about one part or variant, \
or about the audience ("family game") doesn't count. If the card doesn't list complexity, only description \
wording can decide.
- Things to avoid. A game has the feature if its card has the tag, OR the description shows the feature actually \
happening in play. A tag always counts; the description can add a feature but never remove a tag. An empty \
category or mechanic list means no tags, not unknown.
  Tags: fighting = "Fighting" category. Violence = "Wargame" category or the Fighting tag. Horror = "Horror" or \
"Zombies" category. Timer = "Real-time" category or "Real-Time" mechanic. Elimination = "Player Elimination" \
mechanic.
  FIGHTING: players' characters or creatures attack, battle or duel other characters or creatures. NOT fighting: \
armies/fleets/units attacking on a map, raiding or conquering for points (those are violence), metaphors \
("disease-fighting", "fight over resources", "battle to spread plagues"), backstory, contests merely called \
duels or battles, penalties called attacks (Catan's pirate, thieves stealing goods), abstract captures like chess, \
a genre label alone ("dungeon crawler").
  VIOLENCE: the game is about war, battles between armies, raiding, pillaging, conquest or killing, or it has \
fighting. NOT violence: penalties called attacks, metaphors, a grim backstory.
  HORROR: the game is meant to frighten: horror creatures (vampires, zombies, werewolves, Cthulhu monsters) as the \
threat, threatening ghosts or hauntings, being hunted, frightening gore. NOT horror: generic fantasy monsters and \
dungeons, harmless or helpful ghosts, undead or demons as enemies in heroic or comic fantasy, a horror word only in \
a licensed title, realistic threats, cartoon spookiness in family or children's games, one horror character among \
many, giant monsters smashing a city, murder mysteries.
  TIMER: racing a clock or sand timer, everyone playing at once as fast as they can, grabbing or running at the \
same moment, timed turns. NOT a timer: a countdown track that ends the game, simultaneous action choice without \
racing, "fast-paced", racing only as a theme, an optional timer.
  PLAYER ELIMINATION: a player can be knocked out and must sit and watch, for the rest of the game or of a round \
(also when the rules make it unavoidable, e.g. "last person alive wins", or "win all the cards" with 3+ players). \
NOT elimination: losing points or pieces while still playing, a cooperative team losing together, a 2-player \
game that ends when one is knocked out, a knocked-out player who keeps playing in another role.
  A feature counts even if it appears only in an optional variant or one scenario. If the description contradicts \
itself, the more detailed rule wins.
- If a requirement can't be checked from the card or the description, it is UNKNOWN.

STEP 3: Decide.
- A game is ACCEPTABLE only if every requirement FITS. One BREAKS or one UNKNOWN makes it not acceptable.
- Pick an acceptable game. If none is acceptable, pick null. Never pick a game with an UNKNOWN requirement: \
declining is better than a guess.
- In the explanation, say briefly why the game fits, or why none does, using only facts from the card and the \
description."""

# The answer format. Keep it, unless you also change how answer() reads the reply.
FORMAT = """Answer with JSON only, in this form:
{"pick": "<the game's letter, or null to decline>", "explanation": "<one or two sentences for the person>"}"""


def answer(request: Request) -> Answer:
    if not INSTRUCTIONS.strip():
        raise NotImplementedError("Write your instructions in INSTRUCTIONS in systems/single.py first.")
    return call_json(f"{INSTRUCTIONS}\n\n{FORMAT}\n\n{render_request(request)}", Answer)
