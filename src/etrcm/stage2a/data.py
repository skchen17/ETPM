"""Controlled English curriculum; no importance labels or answer-side memory hints."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .language import TOKEN_PATTERN


NAMES = ("alice", "bob", "carol", "david")
DAYS = ("monday", "tuesday", "thursday", "friday")
COLORS = ("red", "blue", "green", "yellow")
PLACES = ("garden", "kitchen", "library", "station")
OBJECTS = ("book", "key", "cup", "map")
FILLERS = (
    "the garden was quiet in the morning .",
    "a small bird waited near the window .",
    "the train arrived after the rain .",
    "they talked about a story at lunch .",
    "the room had a chair and a table .",
)


@dataclass(frozen=True)
class Example:
    text: str
    prefix: str
    answer: str
    kind: str
    chance: float
    gap_tokens: int = 0


def filler(rng: random.Random, minimum_tokens: int) -> str:
    words: list[str] = []
    while len(words) < minimum_tokens:
        words.extend(TOKEN_PATTERN.findall(rng.choice(FILLERS)))
    return " ".join(words[:minimum_tokens])


def sample_basic(rng: random.Random) -> Example:
    name, place, obj, color = rng.choice(NAMES), rng.choice(PLACES), rng.choice(OBJECTS), rng.choice(COLORS)
    text = rng.choice((
        f"{name} walked to the {place} . {name} found a {color} {obj} .",
        f"user : where is {name} ? assistant : {name} is in the {place} .",
        f"the {color} {obj} was on the table . later {name} took it to the {place} .",
        f"{name} and bob talked in the {place} . they saw a {color} {obj} .",
    ))
    return Example(text, "", "", "basic", 0)


def sample_memory(rng: random.Random, gap: int = 16) -> Example:
    if rng.random() < 0.5:
        answer = rng.choice(DAYS)
        statement = f"the meeting is on {answer} ."
        question = "question : when is the meeting ? answer :"
    else:
        answer = rng.choice(NAMES)
        object_name = rng.choice(OBJECTS)
        statement = f"{answer} owns the {object_name} ."
        question = f"question : who owns the {object_name} ? answer :"
    middle = filler(rng, gap)
    prefix = f"{statement} {middle} {question}"
    return Example(f"{prefix} {answer} .", prefix, answer, "memory", 0.25, gap)


def sample_reasoning(rng: random.Random) -> Example:
    kind = rng.choice(("binding", "transitivity", "transition", "relation"))
    if kind == "binding":
        people = list(NAMES[:3]); colors = list(COLORS[:3]); rng.shuffle(people); rng.shuffle(colors)
        selected = rng.randrange(3)
        statements = " ".join(f"{person} has the {color} key ." for person, color in zip(people, colors))
        prefix = f"{statements} question : who has the {colors[selected]} key ? answer :"
        answer = people[selected]; chance = 1 / 3
    elif kind == "transitivity":
        a, b, c = rng.sample(("dax", "wug", "zib", "fep"), 3)
        truth = rng.random() < 0.5
        target = c if truth else rng.choice([x for x in ("dax", "wug", "zib", "fep") if x not in (a,b,c)])
        prefix = f"every {a} is a {b} . every {b} is a {c} . question : is every {a} a {target} ? answer :"
        answer = "yes" if truth else "no"; chance = 0.5
    elif kind == "transition":
        initial = rng.choice(("on", "off")); button = rng.choice(("a", "b"))
        result = "off" if initial == "on" else "on"
        prefix = f"the light is {initial} . pressing button {button} changes the light . button {button} is pressed . question : is the light on or off ? answer :"
        answer = result; chance = 0.5
    else:
        people = rng.sample(NAMES, 3)
        prefix = f"{people[0]} is north of {people[1]} . {people[1]} is north of {people[2]} . question : who is south of {people[0]} ? answer :"
        answer = people[2]; chance = 1 / 3
    return Example(f"{prefix} {answer} .", prefix, answer, kind, chance)


def make_corpus(seed: int, counts: dict[str, int], *, memory_gaps=(0, 8, 16, 32)) -> dict[str, list[Example]]:
    rng = random.Random(seed)
    return {
        "basic": [sample_basic(rng) for _ in range(counts.get("basic", 0))],
        "memory": [sample_memory(rng, rng.choice(memory_gaps)) for _ in range(counts.get("memory", 0))],
        "reasoning": [sample_reasoning(rng) for _ in range(counts.get("reasoning", 0))],
    }
