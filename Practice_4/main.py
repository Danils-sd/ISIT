"""Q-learning: выровнять компот в пяти стаканах с минимумом поднятий."""

import random
from functools import cache
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Сохраняем график без открытия отдельного окна.
import matplotlib.pyplot as plt


N_GLASSES = 5
CAPACITY = 4
TOTAL = 10
TARGET = TOTAL // N_GLASSES
MAX_STEPS = 200
SEED = 42

EPISODES = 500_000
ALPHA = 0.2
GAMMA = 1.0
EPSILON_START = 1.0
EPSILON_END = 0.05
WINDOW = 500
OUTPUT_DIR = Path(__file__).resolve().parent


def is_goal(volumes):
    return all(volume == TARGET for volume in volumes)


@cache
def valid_actions(volumes):
    """Только физические ограничения; невыгодные действия тоже разрешены."""
    return tuple(
        (i, j)
        for i in range(N_GLASSES)
        for j in range(N_GLASSES)
        if i != j and volumes[i] > 0 and volumes[j] < CAPACITY
    )


def step(state, action):
    """Одна порция из i в j; смена источника стоит одного поднятия."""
    if action not in valid_actions(state[:-1]):
        raise ValueError(f"Недопустимое переливание: {state}, {action}")
    i, j = action
    volumes = list(state[:-1])
    volumes[i] -= 1
    volumes[j] += 1
    lifted = int(state[-1] != i)
    done = is_goal(volumes)
    reward = -lifted - 0.001 + (10 if done else 0)
    return (*volumes, i), reward, done, lifted


def optimal_lifts(volumes):
    """Эталон только для проверки, не для выбора действий агента."""
    return sum(volume > TARGET for volume in volumes)


def best_action(q, state, rng=None):
    actions = valid_actions(state[:-1])
    values = [q.get((state, action), 0.0) for action in actions]
    best = max(values)
    tied = [action for action, value in zip(actions, values) if value == best]
    # В проверке выбираем первую пару по порядку, без случайности.
    return rng.choice(tied) if rng is not None else tied[0]


def train(initial_states):
    rng = random.Random(SEED)
    q = {}
    rewards = []
    decay_episodes = int(EPISODES * 0.8)

    for episode in range(EPISODES):
        state = (*rng.choice(initial_states), -1)
        fraction = min(episode / max(decay_episodes - 1, 1), 1.0)
        epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * fraction
        total_reward = 0.0

        for turn in range(MAX_STEPS):
            if rng.random() < epsilon:
                action = rng.choice(valid_actions(state[:-1]))
            else:
                action = best_action(q, state, rng)
            next_state, reward, done, _ = step(state, action)
            truncated = turn == MAX_STEPS - 1 and not done
            if truncated:
                reward -= 10

            future = 0.0
            if not done and not truncated:
                future = max(
                    q.get((next_state, a), 0.0)
                    for a in valid_actions(next_state[:-1])
                )
            key = (state, action)
            old_value = q.get(key, 0.0)
            q[key] = old_value + ALPHA * (reward + GAMMA * future - old_value)
            total_reward += reward
            state = next_state
            if done or truncated:
                break

        rewards.append(total_reward)
        if (episode + 1) % 10_000 == 0:
            average = sum(rewards[-1000:]) / 1000
            print(
                f"Эпизод {episode + 1}/{EPISODES}: "
                f"epsilon={epsilon:.3f}, средняя награда={average:.3f}",
                flush=True,
            )
    return q, rewards


def rollout(q, volumes):
    """Проверка без обучения, исследования и изменения Q-таблицы."""
    state = (*volumes, -1)
    lifts = 0
    trace = []
    for _ in range(MAX_STEPS):
        if is_goal(state[:-1]):
            break
        action = best_action(q, state)
        state, _, _, lifted = step(state, action)
        lifts += lifted
        trace.append((action, state[:-1], lifts))
    return is_goal(state[:-1]), lifts, trace


def check_environment(all_states):
    assert TOTAL % N_GLASSES == 0 and 0 < TARGET <= CAPACITY
    assert len(all_states) == 381
    # Проверяем сохранение жидкости и вместимость для всех допустимых действий.
    for volumes in all_states:
        for action in valid_actions(volumes):
            state, _, _, lifted = step((*volumes, -1), action)
            assert sum(state[:-1]) == TOTAL
            assert all(0 <= value <= CAPACITY for value in state[:-1])
            assert lifted == 1

    state = (4, 3, 2, 1, 0, -1)
    state, _, done, lifted = step(state, (0, 3))
    assert lifted == 1 and not done
    state, _, done, lifted = step(state, (0, 4))
    assert lifted == 0 and not done  # Другой приёмник, тот же источник.
    state, reward, done, lifted = step(state, (1, 4))
    assert lifted == 1 and done and abs(reward - 8.999) < 1e-9
    assert state[:-1] == (2, 2, 2, 2, 2)
    assert optimal_lifts((4, 3, 2, 1, 0)) == 2

    state, _, _, _ = step((4, 3, 2, 1, 0, -1), (0, 3))
    state, _, _, _ = step(state, (1, 4))
    _, _, _, lifted = step(state, (0, 4))
    assert lifted == 1  # Вернулись к источнику после смены стакана.
    assert rollout({}, (2, 2, 2, 2, 2)) == (True, 0, [])

    for invalid in ((0, 0), (4, 3), (1, 0)):
        try:
            step((4, 3, 2, 1, 0, -1), invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Разрешено недопустимое действие {invalid}")


def save_results(q, initial_states, rewards):
    results = []
    table_size = len(q)
    for volumes in initial_states:
        success, lifts, trace = rollout(q, volumes)
        minimum = optimal_lifts(volumes)
        results.append((volumes, success, lifts, len(trace), minimum))
    assert len(q) == table_size
    assert rollout(q, (2, 2, 2, 2, 2)) == (True, 0, [])

    count = len(results)
    solved = sum(success for _, success, _, _, _ in results)
    optimal = sum(success and lifts == minimum
                  for _, success, lifts, _, minimum in results)
    summary = (
        f"Эпизодов обучения: {EPISODES}; seed: {SEED}\n"
        f"alpha={ALPHA}, gamma={GAMMA}, epsilon: {EPSILON_START} -> {EPSILON_END}\n"
        f"Записей Q-таблицы: {len(q)}\n"
        f"Средняя награда первых 1000 эпизодов: {sum(rewards[:1000]) / 1000:.3f}\n"
        f"Средняя награда последних 1000 эпизодов: {sum(rewards[-1000:]) / 1000:.3f}\n"
        f"Проверено неравных распределений: {count}\n"
        f"Успешно: {solved}/{count} ({solved / count:.2%})\n"
        f"Оптимально по поднятиям: {optimal}/{count} ({optimal / count:.2%})\n"
        "Равное распределение: успех без действий и поднятий.\n"
        "Проверка: epsilon=0, фиксированный порядок при равных Q-оценках.\n"
        "Все проверяемые распределения входят в область обучения; "
        "обобщение на другие объёмы не проверяется.\n"
    )
    lines = [summary, "Результат для каждого начального распределения:",
             "Объёмы | Успех | Поднятия | Переливания | Минимум | Оптимально"]
    for volumes, success, lifts, turns, minimum in results:
        lines.append(
            f"{volumes} | {success} | {lifts} | {turns} | {minimum} | "
            f"{success and lifts == minimum}"
        )

    examples = [(4, 3, 2, 1, 0), (4, 4, 2, 0, 0),
                (3, 3, 3, 1, 0), (2, 2, 2, 2, 2)]
    for volumes in examples:
        success, lifts, trace = rollout(q, volumes)
        lines.append(f"\nПример: {volumes}; минимум поднятий: {optimal_lifts(volumes)}")
        for turn, ((i, j), new_volumes, lifted) in enumerate(trace, 1):
            lines.append(
                f"  {turn}. Стакан {i + 1} -> {j + 1}: {new_volumes}; "
                f"поднятий: {lifted}"
            )
        lines.append(f"  Успех: {success}; поднятий: {lifts}; переливаний: {len(trace)}")

    path = OUTPUT_DIR / "results.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + summary, flush=True)
    return solved == count and optimal == count


def plot_learning(rewards):
    average = []
    window_sum = 0.0
    for index, reward in enumerate(rewards):
        window_sum += reward
        if index >= WINDOW:
            window_sum -= rewards[index - WINDOW]
        if index >= WINDOW - 1:
            average.append(window_sum / WINDOW)

    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    ax.plot(range(1, len(rewards) + 1), rewards, color="#607d9c",
            alpha=0.2, linewidth=0.5, label="Награда за эпизод")
    ax.plot(range(WINDOW, len(rewards) + 1), average, color="#c44724",
            linewidth=1.7, label=f"Среднее за {WINDOW} эпизодов")
    ax.set(title="Q-learning: выравнивание компота в пяти стаканах",
           xlabel="Эпизод обучения", ylabel="Суммарная награда")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(OUTPUT_DIR / "learning_curve.png", dpi=160)
    plt.close(fig)


def main():
    all_states = [volumes for volumes in product(range(CAPACITY + 1), repeat=N_GLASSES)
                  if sum(volumes) == TOTAL]
    check_environment(all_states)
    print("Проверки среды пройдены.", flush=True)
    initial_states = [volumes for volumes in all_states if not is_goal(volumes)]
    q, rewards = train(initial_states)
    passed = save_results(q, initial_states, rewards)
    plot_learning(rewards)
    print(f"График и результаты сохранены в {OUTPUT_DIR}", flush=True)
    if not passed:
        raise SystemExit("Проверка оптимальности не пройдена. См. results.txt.")


if __name__ == "__main__":
    main()
