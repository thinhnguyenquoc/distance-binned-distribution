import sys

file_path = 'od_plan_tester/tests/test_experiment_contracts.py'
with open(file_path, 'r') as f:
    content = f.read()

old_call = """    res = run_target_city_experiments(
        model=model,
        city_name="Denver",
        scaler=fitted_scaler,
        data_root="data",
        num_trip_seeds=2,
        m_grid=[100, 1000],
        device_str="cpu",
    )"""

new_call = """    from src.experiment.run_experiment import compute_kbin_edges
    bin_edges, _ = compute_kbin_edges(["Raleigh", "Denver"], K=8, data_root="data")
    res = run_target_city_experiments(
        model=model,
        city_name="Denver",
        scaler=fitted_scaler,
        data_root="data",
        bin_edges=bin_edges,
        device_str="cpu",
    )"""

content = content.replace(old_call, new_call)
with open(file_path, 'w') as f:
    f.write(content)
