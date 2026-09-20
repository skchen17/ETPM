"""Counterbalanced synthetic English associations without memory-key labels."""

from __future__ import annotations

import random
from dataclasses import dataclass

from etrcm.stage2a.language import TOKEN_PATTERN


NAMES = ("alice", "bob", "carol", "david", "emma", "frank", "grace", "henry",
         "iris", "jack", "kate", "liam", "maya", "noah", "olivia", "peter",
         "quinn", "rose", "sam", "tina", "uma", "victor", "wendy", "xavier",
         "yasmin", "zack", "amelia", "bruno", "clara", "diego", "elena", "felix")
COLORS = ("red", "blue", "green", "yellow", "purple", "orange", "white", "black")
PLACES = ("library", "garden", "station", "kitchen", "museum", "office", "market", "school")
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today")
OBJECTS = ("key", "map", "cup", "book")
FILLER = ("the wind was quiet near the door .", "a bird waited by the window .",
          "they walked along the road after lunch .", "the room had a table and a chair .",
          "a small train passed the old bridge .")
FAMILIES = ("attribute", "location", "revision", "relation", "temporal", "interference")


@dataclass(frozen=True)
class Association:
    text: str
    prefix: str
    answer: str
    candidates: tuple[str, ...]
    family: str
    gap: int
    split: str
    entity: str
    value: str
    question: str


@dataclass(frozen=True)
class SimpleText:
    text: str
    prefix: str = ""
    answer: str = ""
    kind: str = "basic"


def sample_basic_safe(rng: random.Random) -> SimpleText:
    """Basic English without any name+attribute/location co-occurrence."""
    name, color, place = rng.choice(NAMES),rng.choice(COLORS),rng.choice(PLACES)
    return SimpleText(rng.choice((
        f"{name} walked slowly . {name} smiled .",
        f"the {color} key was on the table .",
        f"the {place} was quiet in the morning .",
        "user : hello . assistant : hello .",
    )))


def token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def filler(rng: random.Random, gap: int) -> str:
    tokens: list[str] = []
    while len(tokens) < gap:
        tokens += TOKEN_PATTERN.findall(rng.choice(FILLER))
    return " ".join(tokens[:gap])


def allowed(name: str, value: str, values: tuple[str, ...], split: str) -> bool:
    """Each lexical item occurs in train; entity-value pair parity is disjoint."""
    parity = (NAMES.index(name) + values.index(value)) % 2
    return parity == (0 if split == "train" else 1)


def pick_value(rng: random.Random, name: str, values: tuple[str, ...], split: str) -> str:
    return rng.choice([v for v in values if allowed(name, v, values, split)])


def make_association(rng: random.Random, family: str, gap: int, split: str,
                     *, entity_count: int = 8) -> Association:
    if family not in FAMILIES or split not in {"train", "ood"}:
        raise ValueError((family, split))
    people = rng.sample(NAMES, max(2, entity_count if family == "interference" else 2))
    person = people[0]
    other = people[1]
    if family == "relation":
        desired = 0 if split == "train" else 1
        other = rng.choice([n for n in NAMES if n != person and (NAMES.index(person) + NAMES.index(n)) % 2 == desired])
    if family == "attribute":
        value = pick_value(rng, person, COLORS, split)
        other_value = pick_value(rng, other, COLORS, split)
        order = [(person, value), (other, other_value)]
        rng.shuffle(order)
        premise = " ".join(f"{name} has the {color} key ." for name, color in order)
        question = f"question : what key does {person} have ? answer :"
        candidates = COLORS
    elif family == "location":
        value = pick_value(rng, person, PLACES, split)
        other_value = pick_value(rng, other, PLACES, split)
        order = [(person, value), (other, other_value)]
        rng.shuffle(order)
        premise = " ".join(f"{name} went to the {place} ." for name, place in order)
        question = f"question : where is {person} ? answer :"
        candidates = PLACES
    elif family == "revision":
        value = pick_value(rng, person, COLORS, split)
        old = rng.choice([v for v in COLORS if v != value and allowed(person, v, COLORS, split)])
        premise = f"{person} has the {old} key . {other} has the {value} map . "
        premise += f"{person} gives away the {old} key . {person} receives the {value} key ."
        question = f"question : what key does {person} have now ? answer :"
        candidates = COLORS
    elif family == "relation":
        value = rng.choice(OBJECTS)
        premise = f"{person} has the {value} . {person} gave the {value} to {other} ."
        question = f"question : who has the {value} now ? answer :"
        value = other
        candidates = NAMES
    elif family == "temporal":
        # Day transitions are balanced; `person` is a lexical context cue only.
        value = pick_value(rng, person, DAYS, split)
        old = rng.choice([v for v in DAYS if v != value])
        premise = f"{person} said the meeting is on {old} . the meeting was moved to {value} ."
        question = "question : when is the meeting ? answer :"
        candidates = DAYS
    else:
        value = pick_value(rng, person, COLORS, split)
        rows = [(person, value)] + [(name, pick_value(rng, name, COLORS, split)) for name in people[1:]]
        rng.shuffle(rows)
        premise = " ".join(f"{name} has the {color} key ." for name, color in rows)
        question = f"question : what key does {person} have ? answer :"
        candidates = COLORS
    prefix = f"{premise} {filler(rng, gap)} {question}"
    return Association(f"{prefix} {value} .", prefix, value, tuple(candidates),
                       family, gap, split, person, value, question)


def counterfactual_pair(rng: random.Random, gap: int = 32) -> tuple[Association, Association]:
    """Same bag of words, different entity->color relation, controlled position."""
    a, b = rng.sample(NAMES, 2)
    c, d = rng.sample(COLORS, 2)
    middle = filler(rng, gap)
    question = f"question : what key does {a} have ? answer :"
    if rng.random() < 0.5:
        order_a, order_b = ((a,c),(b,d)), ((a,d),(b,c))
    else:
        order_a, order_b = ((b,d),(a,c)), ((b,c),(a,d))
    prefix_a = " ".join(f"{name} has the {color} key ." for name,color in order_a) + f" {middle} {question}"
    prefix_b = " ".join(f"{name} has the {color} key ." for name,color in order_b) + f" {middle} {question}"
    return (
        Association(f"{prefix_a} {c} .", prefix_a, c, (c,d), "counterfactual", gap, "paired", a, c, question),
        Association(f"{prefix_b} {d} .", prefix_b, d, (c,d), "counterfactual", gap, "paired", a, d, question),
    )


def make_corpus(seed: int, count: int, split: str, gaps=(16, 32, 64, 128)) -> list[Association]:
    rng = random.Random(seed)
    examples = []
    for i in range(count):
        family = FAMILIES[i % len(FAMILIES)]
        gap = gaps[(i // len(FAMILIES)) % len(gaps)]
        entity_count = (8,16,32)[(i // len(FAMILIES)) % 3] if family == "interference" else 2
        # Avoid >400-token training examples; 32-way interference is evaluated
        # at 16-token gap, while 8-way cases cover all trained gaps.
        if family == "interference" and entity_count >= 16:
            gap = 16
        examples.append(make_association(rng, family, gap, split, entity_count=entity_count))
    rng.shuffle(examples)
    return examples
