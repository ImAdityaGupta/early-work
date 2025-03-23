# MonsterChess

**Monster Chess** is a lesser-known variant of chess for which I intermittently developed a minimax-inspired engine between 2019 and 2022.

---

## Project Structure

### `initial_try`

- Very early prototype code.
- Basic functionality, no UI.

### `Chess_after_tutorials_1`

- A much more advanced version with a graphical user interface built using Qt 5.12 and MinGW.
- In theory, it should run on any Windows machine after setup.

---
## Project Brief

My project involves building a chess engine to play a variant of chess called **Monster Chess**.

There are many variants of chess — chess-like games with altered rules. These altered rules lead to dramatically different play. Monster Chess was introduced to me by my father when I was very young, so this project combines my love for programming and chess with the desire to beat my father at the game he taught me.

Monster Chess differs from standard chess in two crucial ways:

- White gets two moves for every one move Black gets.
- Black starts with their full set of pieces, while White starts with only a king and four pawns.

The power of the double-move king is what gives Monster Chess its name — the Monster White King.

---

## How the Engine Works

My engine consists of two core components:

- **Move Generator**: Modifies traditional chess move generation to produce legal moves for Monster Chess.
- **Move Selector**: Chooses what the engine thinks is the best move to play.

The engine is built in C++. I also created a User Interface using Qt so that non-programmers can easily play against the engine.

---

## Background

Monster Chess began as a way to be productive during the extra free time I had during the COVID-19 lockdowns in 2021, though I had been tinkering on it before then. When schools reopened, I realized I had a prime opportunity to explore the topic in more depth by doing an EPQ (Extended Project Qualification) around it.

As a result, some of the evaluation and logging were done out of order or retrospectively.

More details on the rules, implementation, and approach can be found in:

- [Research Writeup (PDF)](Monster_Chess_Research_Writeup.pdf)
- [Chess_after_tutorials_1 folder](Chess_after_tutorials_1/)

---

