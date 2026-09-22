import numpy as np
import skfuzzy as fuzz

def recommend_wash(load_kg: float, dirt_level: float, water_saving: float):
    # Input universes
    load_x = np.arange(0, 11, 0.1)
    dirt_x = np.arange(0, 101, 1)
    save_x = np.arange(0, 101, 1)
    output_x = np.arange(0, 101, 1)

    # Membership functions
    load_small = fuzz.trimf(load_x, [0, 0, 4])
    load_medium = fuzz.trimf(load_x, [2, 5, 8])
    load_large = fuzz.trimf(load_x, [6, 10, 10])

    dirt_low = fuzz.trimf(dirt_x, [0, 0, 35])
    dirt_medium = fuzz.trimf(dirt_x, [20, 50, 80])
    dirt_high = fuzz.trimf(dirt_x, [60, 100, 100])

    save_low = fuzz.trimf(save_x, [0, 0, 40])
    save_medium = fuzz.trimf(save_x, [20, 50, 80])
    save_high = fuzz.trimf(save_x, [60, 100, 100])

    gentle = fuzz.trimf(output_x, [0, 0, 40])
    normal = fuzz.trimf(output_x, [25, 50, 75])
    heavy = fuzz.trimf(output_x, [60, 100, 100])

    # Fuzzification
    load_membership = {
        "small": float(fuzz.interp_membership(load_x, load_small, load_kg)),
        "medium": float(fuzz.interp_membership(load_x, load_medium, load_kg)),
        "large": float(fuzz.interp_membership(load_x, load_large, load_kg)),
    }
    dirt_membership = {
        "low": float(fuzz.interp_membership(dirt_x, dirt_low, dirt_level)),
        "medium": float(fuzz.interp_membership(dirt_x, dirt_medium, dirt_level)),
        "high": float(fuzz.interp_membership(dirt_x, dirt_high, dirt_level)),
    }
    save_membership = {
        "low": float(fuzz.interp_membership(save_x, save_low, water_saving)),
        "medium": float(fuzz.interp_membership(save_x, save_medium, water_saving)),
        "high": float(fuzz.interp_membership(save_x, save_high, water_saving)),
    }

    # Rule evaluation (Mamdani-style max-min inference)
    rules = [
        ("R1: dirt high AND load large -> heavy",
         min(dirt_membership["high"], load_membership["large"]), "heavy"),
        ("R2: dirt high AND load medium -> heavy",
         min(dirt_membership["high"], load_membership["medium"]), "heavy"),
        ("R3: dirt medium AND load large -> heavy",
         min(dirt_membership["medium"], load_membership["large"]), "heavy"),
        ("R4: dirt medium AND load medium -> normal",
         min(dirt_membership["medium"], load_membership["medium"]), "normal"),
        ("R5: dirt low AND load small -> gentle",
         min(dirt_membership["low"], load_membership["small"]), "gentle"),
        ("R6: dirt low AND load medium -> gentle",
         min(dirt_membership["low"], load_membership["medium"]), "gentle"),
        ("R7: water saving high AND dirt low -> gentle",
         min(save_membership["high"], dirt_membership["low"]), "gentle"),
        ("R8: water saving high AND dirt medium -> normal",
         min(save_membership["high"], dirt_membership["medium"]), "normal"),
        ("R9: water saving low AND dirt high -> heavy",
         min(save_membership["low"], dirt_membership["high"]), "heavy"),
        ("R10: water saving medium AND dirt medium -> normal",
         min(save_membership["medium"], dirt_membership["medium"]), "normal"),
    ]

    rule_activations = {name: round(strength, 3) for name, strength, _ in rules}

    # Aggregate output membership
    aggregated = np.zeros_like(output_x, dtype=float)
    for _, strength, output_name in rules:
        output_mf = {"gentle": gentle, "normal": normal, "heavy": heavy}[output_name]
        aggregated = np.fmax(aggregated, np.fmin(strength, output_mf))

    # Defuzzification
    if np.sum(aggregated) == 0:
        crisp = 50.0
    else:
        crisp = float(fuzz.defuzz(output_x, aggregated, "centroid"))

    if crisp < 35:
        intensity = "Gentle"
        wash_time = 25
        water_level = "Low"
        spin = "Low"
    elif crisp < 68:
        intensity = "Normal"
        wash_time = 35
        water_level = "Medium"
        spin = "Medium"
    else:
        intensity = "Heavy"
        wash_time = 45
        water_level = "High"
        spin = "Medium"

    return {
        "wash_intensity": intensity,
        "wash_time": wash_time,
        "water_level": water_level,
        "spin": spin,
        "crisp_intensity": crisp,
        "memberships": {
            "load": {k: round(v, 3) for k, v in load_membership.items()},
            "dirt": {k: round(v, 3) for k, v in dirt_membership.items()},
            "water_saving": {k: round(v, 3) for k, v in save_membership.items()},
        },
        "rule_activations": rule_activations,
    }
