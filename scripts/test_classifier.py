def classify_material(props):
    E11, E22, G12, nu12 = props['E11'], props['E22'], props['G12'], props['nu12']
    r = E11 / max(E22, 1e-6)
    s = G12 / max(E11, 1e-6)
    if r < 1.15 and E11 < 20.0 and nu12 > 0.25:
        return 'Random / CSM'
    if r < 1.15 and s < 0.15 and nu12 > 0.25:
        return 'Quasi-Isotropic'
    if r < 1.20 and s >= 0.15 and nu12 <= 0.20:
        return 'Woven Fabric'
    if r < 1.20:
        return 'Quasi-Isotropic'
    if 1.20 <= r < 6.0 and s > 0.20:
        return 'Angle-Ply [+/-th]'
    if 1.20 <= r < 5.5 and s <= 0.20:
        return 'Cross-Ply [0/90]'
    if r >= 5.5 and nu12 >= 0.20:
        return 'Unidirectional (UD)'
    return 'Multi-Directional'

tests = [
    ('Your specimen (Woven)',     {'E11':19.7, 'E22':20.3, 'G12':4.50, 'nu12':0.11}, 'Woven Fabric'),
    ('Woven Glass/Epoxy',        {'E11':25.0, 'E22':24.0, 'G12':4.00, 'nu12':0.13}, 'Woven Fabric'),
    ('Quasi-Isotropic CF/EP',    {'E11':55.0, 'E22':55.0, 'G12':21.0, 'nu12':0.31}, 'Quasi-Isotropic'),
    ('Random CSM Glass',         {'E11': 8.5, 'E22': 8.5, 'G12':3.00, 'nu12':0.32}, 'Random / CSM'),
    ('Cross-Ply [0/90] UD-CF',   {'E11':80.0, 'E22':40.0, 'G12':5.00, 'nu12':0.08}, 'Cross-Ply [0/90]'),
    ('Angle-Ply [+/-45] CF/EP',  {'E11':20.0, 'E22':15.0, 'G12':30.0, 'nu12':0.60}, 'Angle-Ply [+/-th]'),
    ('Unidirectional CF/EP',     {'E11':140., 'E22': 9.0, 'G12':5.00, 'nu12':0.30}, 'Unidirectional (UD)'),
    ('Unidirectional Glass/EP',  {'E11': 45., 'E22': 8.0, 'G12':4.50, 'nu12':0.28}, 'Unidirectional (UD)'),
]

print('=' * 80)
print(f"  {'Material':<28}  r      s       nu12   ->  Got              Expected")
print('=' * 80)
all_pass = True
for label, p, expected in tests:
    r = p['E11'] / p['E22']
    s = p['G12'] / p['E11']
    result = classify_material(p)
    ok = 'PASS' if result == expected else 'FAIL'
    if ok == 'FAIL':
        all_pass = False
    print(f"  {label:<28}  {r:.3f}  {s:.4f}  {p['nu12']:.2f}   ->  {result:<20} [{ok}]")
print('=' * 80)
print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES - check above'}")
print('=' * 80)
