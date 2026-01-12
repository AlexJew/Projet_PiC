# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "anthropic==0.75.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "mcp==1.24.0",
#     "numpy==2.3.5",
#     "openpyxl==3.1.5",
#     "pandas==2.3.3",
#     "pyarrow==22.0.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Library import, configurations and generic functions
    This first section focuses on the import of libraries and the definition of the configurations, as well as the definition of recurrent functions.
    """)
    return


@app.cell(hide_code=True)
def library_and_config():
    from dataclasses import dataclass, field
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import altair as alt

    @dataclass(frozen=True)
    class PVConfig:
        reference_panels: int = 21
        panel_power: float = 340.0  # W

    @dataclass(frozen=True)
    class BatteryConfig:
        efficiency: float = 0.95  # dimensionless
        initial_soc: float = 0.5  # dimensionless
        min_soc: float = 0.1  # dimensionless
        max_soc: float = 0.9  # dimensionless

    @dataclass(frozen=True)
    class AnalysisConfig:
        pv_range: np.ndarray = field(default_factory=lambda: np.arange(1, 51, 1))  # dimensionless
        battery_power_range: np.ndarray = field(default_factory=lambda: np.arange(1000, 100001, 1000))  # W

    @dataclass(frozen=True)
    class OptimizationConfig:
        default_weight: float = 0.5  # dimensionless

    @dataclass(frozen=True)
    class PathsConfig:
        data_dir: Path = Path("data")
        output_dir: Path = Path("output")

    @dataclass(frozen=True)
    class PlotConfig:
        line_width: float = 1.5  # pt
        font_size: int = 11  # pt
        summer_color: tuple[float, float, float] = (1.0, 0.0, 0.0)  # dimensionless (red)
        winter_color: tuple[float, float, float] = (0.0, 0.5, 1.0)  # dimensionless (blue)

        # Time axis configuration
        time_unit_divisor: int = 60  # Convert minutes to hours
        time_axis_title: str = "Temps (heures)"  # Axis label with unit
        time_tick_interval: int = 20  # Ticks every 20 hours

    # Instantiate configurations
    pv_config = PVConfig()
    battery_config = BatteryConfig()
    analysis_config = AnalysisConfig()
    optimization_config = OptimizationConfig()
    paths_config = PathsConfig()
    plot_config = PlotConfig()
    return (
        PlotConfig,
        alt,
        battery_config,
        np,
        paths_config,
        pd,
        plot_config,
        plt,
        pv_config,
    )


@app.cell(hide_code=True)
def _(PlotConfig, alt):
    def create_time_axis_encoding(plot_config: PlotConfig, max_hours: int = 168) -> alt.X:
        # Calculate tick positions in minutes for clean hour intervals
        # Example: For 20-hour intervals: 0, 1200, 2400, 3600, ... minutes
        tick_interval_minutes = plot_config.time_tick_interval * plot_config.time_unit_divisor
        max_minutes = max_hours * plot_config.time_unit_divisor
        tick_values = list(range(0, max_minutes + 1, tick_interval_minutes))

        return alt.X(
            'Index:Q',
            title=plot_config.time_axis_title,
            axis=alt.Axis(
                values=tick_values,  # Explicit tick positions in minutes
                labelExpr=f'datum.value / {plot_config.time_unit_divisor}',
                format='.0f',  # Whole numbers only
                grid=True
            )
        )

    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
    return create_time_axis_encoding, rgb_to_hex


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Import
    This section handles the import of Excel files from the data directory as pandas DataFrames. It displays the load at the end.
    """)
    return


@app.cell
def data_import(paths_config, pd):

    # Define data directory path
    data_dir = paths_config.data_dir

    # Import housing consumption data (Logement_1 to Logement_6)
    logement_1 = pd.read_excel(data_dir / "Logement_1.xlsx")
    logement_2 = pd.read_excel(data_dir / "Logement_2.xlsx")
    logement_3 = pd.read_excel(data_dir / "Logement_3.xlsx")
    logement_4 = pd.read_excel(data_dir / "Logement_4.xlsx")
    logement_5 = pd.read_excel(data_dir / "Logement_5.xlsx")
    logement_6 = pd.read_excel(data_dir / "Logement_6.xlsx")

    # Sum all logements to create total charge DataFrame
    # Assuming all logements have the same structure with numeric columns to sum
    charge = logement_1.copy()
    _numeric_cols = charge.select_dtypes(include=['float64', 'int64']).columns

    for col in _numeric_cols:
        charge[col] = (
            logement_1[col].fillna(0) +
            logement_2[col].fillna(0) +
            logement_3[col].fillna(0) +
            logement_4[col].fillna(0) +
            logement_5[col].fillna(0) +
            logement_6[col].fillna(0)
        )

    # Rename columns to 'Été' and 'Hiver' for plotting
    # Assuming the first two numeric columns are summer and winter data
    if len(_numeric_cols) >= 2:
        column_mapping = {
            _numeric_cols[0]: 'Hiver',
            _numeric_cols[1]: 'Été'
        }
        charge = charge.rename(columns=column_mapping)

    # Import solar production data
    solaire_ete = pd.read_excel(data_dir / "Solaire_semaine_été.xlsx")
    solaire_hiver = pd.read_excel(data_dir / "Solaire_semaine_hiver.xlsx")

    # Import tariff data
    tarif_hchp = pd.read_excel(data_dir / "Tarif_HCHP.xlsx")

    # Display the charge dataframe
    charge
    return charge, solaire_ete, solaire_hiver


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modélisation charge + PV simple
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Données de consommation
    Les données de consommation des 6 bâtiments en hiver et en été sont visualisées
    """)
    return


@app.cell(hide_code=True)
def consumption_data_plot(
    alt,
    charge,
    create_time_axis_encoding,
    mo,
    pd,
    plot_config,
    rgb_to_hex,
):

    # Prepare data for Altair (long format)
    charge_long = pd.DataFrame({
        'Index': range(len(charge)),
        'Été': charge['Été'].values,
        'Hiver': charge['Hiver'].values
    }).melt(id_vars=['Index'], var_name='Saison', value_name='Consommation')

    # Create Altair chart
    _chart = alt.Chart(charge_long).mark_line(
        strokeWidth=plot_config.line_width
    ).transform_calculate(
        ConsommationKW='datum.Consommation / 1000'
    ).encode(
        x=create_time_axis_encoding(plot_config),
        y=alt.Y('ConsommationKW:Q', title='Consommation électrique (kW)'),
        color=alt.Color('Saison:N',
                       scale=alt.Scale(
                           domain=['Été', 'Hiver'],
                           range=[rgb_to_hex(plot_config.summer_color),
                                  rgb_to_hex(plot_config.winter_color)]
                       ),
                       legend=alt.Legend(title=None))
    ).properties(
        width=800,
        height=400,
        title='Consommation électrique - Semaine été et hiver'
    ).configure_axis(
        labelFontSize=plot_config.font_size,
        titleFontSize=plot_config.font_size,
        grid=True,
        gridOpacity=0.3
    ).configure_title(
        fontSize=plot_config.font_size + 2
    ).configure_legend(
        labelFontSize=plot_config.font_size
    )

    # Calculate statistics (convert to kW)
    ete_max = charge['Été'].max() / 1000
    ete_avg = charge['Été'].mean() / 1000
    hiver_max = charge['Hiver'].max() / 1000
    hiver_avg = charge['Hiver'].mean() / 1000

    # Create statistics display
    stats_display = mo.md(f"""
    **Statistiques de consommation:**
    - **Été**: Maximum = {ete_max:.2f} kW | Moyenne = {ete_avg:.2f} kW
    - **Hiver**: Maximum = {hiver_max:.2f} kW | Moyenne = {hiver_avg:.2f} kW
    """)

    # Create interactive marimo chart
    chart = mo.ui.altair_chart(_chart)

    # Display statistics and chart together
    mo.vstack([stats_display, chart])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Données de production
    Les données de production solaire pour un panneau en hiver et en été sont visualisées
    """)
    return


@app.cell(hide_code=True)
def _(
    alt,
    create_time_axis_encoding,
    mo,
    pd,
    plot_config,
    pv_config,
    rgb_to_hex,
    solaire_ete,
    solaire_hiver,
):
    # Extract production data from solar DataFrames
    # Get the first numeric column (production values)
    solar_ete_numeric_cols = solaire_ete.select_dtypes(include=['float64', 'int64']).columns
    solar_hiver_numeric_cols = solaire_hiver.select_dtypes(include=['float64', 'int64']).columns

    if len(solar_ete_numeric_cols) > 0:
        production_ete_reference = solaire_ete[solar_ete_numeric_cols[0]].values
    else:
        production_ete_reference = solaire_ete.iloc[:, 0].values
    if len(solar_hiver_numeric_cols) > 0:
        production_hiver_reference = solaire_hiver[solar_hiver_numeric_cols[0]].values
    else:
        production_hiver_reference = solaire_hiver.iloc[:, 0].values

    # Scale to single panel production
    production_ete_single = production_ete_reference / pv_config.reference_panels
    production_hiver_single = production_hiver_reference / pv_config.reference_panels

    # Ensure both arrays have the same length for plotting
    min_length_prod = min(len(production_ete_single), len(production_hiver_single))
    production_ete_single = production_ete_single[:min_length_prod]
    production_hiver_single = production_hiver_single[:min_length_prod]

    # Prepare data for Altair (long format)
    production_long = pd.DataFrame({
        'Index': range(min_length_prod),
        'Été': production_ete_single,
        'Hiver': production_hiver_single
    }).melt(id_vars=['Index'], var_name='Saison', value_name='Production')

    # Create Altair chart
    _chart_prod = alt.Chart(production_long).mark_line(
        strokeWidth=plot_config.line_width
    ).encode(
        x=create_time_axis_encoding(plot_config),
        y=alt.Y('Production:Q', title='Production solaire (W)'),
        color=alt.Color('Saison:N',
                       scale=alt.Scale(
                           domain=['Été', 'Hiver'],
                           range=[rgb_to_hex(plot_config.summer_color),
                                  rgb_to_hex(plot_config.winter_color)]
                       ),
                       legend=alt.Legend(title=None))
    ).properties(
        width=800,
        height=400,
        title='Production solaire - Semaine été et hiver (1 panneau)'
    ).configure_axis(
        labelFontSize=plot_config.font_size,
        titleFontSize=plot_config.font_size,
        grid=True,
        gridOpacity=0.3
    ).configure_title(
        fontSize=plot_config.font_size + 2
    ).configure_legend(
        labelFontSize=plot_config.font_size
    )

    # Calculate statistics
    ete_max_prod = production_ete_single.max()
    ete_avg_prod = production_ete_single.mean()
    hiver_max_prod = production_hiver_single.max()
    hiver_avg_prod = production_hiver_single.mean()

    # Create statistics display
    stats_display_prod = mo.md(f"""
    **Statistiques de production (1 panneau):**
    - **Été**: Maximum = {ete_max_prod:.2f} W | Moyenne = {ete_avg_prod:.2f} W
    - **Hiver**: Maximum = {hiver_max_prod:.2f} W | Moyenne = {hiver_avg_prod:.2f} W
    """)

    # Create interactive marimo chart
    chart_prod = mo.ui.altair_chart(_chart_prod)

    # Display statistics and chart together
    mo.vstack([stats_display_prod, chart_prod])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Puissance nette en été
    Visualisation de la charge nette (production - consommation solaire) en été pour différents nombres de panneaux PV
    """)
    return


@app.cell(hide_code=True)
def _(
    alt,
    charge,
    create_time_axis_encoding,
    mo,
    pd,
    plot_config,
    pv_config,
    solaire_ete,
):

    # Extract summer consumption - assuming it's the Été column
    consumption_ete = charge['Été'].values

    # Get the first numeric column from solar data (production values)
    solar_numeric_cols = solaire_ete.select_dtypes(include=['float64', 'int64']).columns
    if len(solar_numeric_cols) > 0:
        # Use the first numeric column as production data
        production_reference = solaire_ete[solar_numeric_cols[0]].values
    else:
        # Fallback if no numeric columns found
        production_reference = solaire_ete.iloc[:, 0].values

    # Ensure both arrays have the same length
    min_length = min(len(consumption_ete), len(production_reference))
    consumption_ete = consumption_ete[:min_length]
    production_reference = production_reference[:min_length]

    # Panel counts to analyze
    panel_counts = [5, 10, 20, 50]

    # Calculate net load for each panel count
    net_production_data = {'Index': range(min_length)}

    for _n_panels in panel_counts:
        # Scale production based on panel count
        _scaling_factor = _n_panels / pv_config.reference_panels
        production_scaled = production_reference * _scaling_factor

        # Calculate net load (consumption - production)
        net_production = production_scaled - consumption_ete

        net_production_data[f'{_n_panels} panneaux'] = net_production

    # Create DataFrame in long format for Altair
    net_production_df = pd.DataFrame(net_production_data)
    net_production_long = net_production_df.melt(
        id_vars=['Index'],
        var_name='Configuration',
        value_name='Charge nette'
    )

    # Create Altair chart (without configure methods)
    _net_chart = alt.Chart(net_production_long).mark_line(
        strokeWidth=plot_config.line_width
    ).transform_calculate(
        ChargeNetteKW='datum["Charge nette"] / 1000'
    ).encode(
        x=create_time_axis_encoding(plot_config),
        y=alt.Y('ChargeNetteKW:Q', title='Puissance nette (kW)'),
        color=alt.Color('Configuration:N',
                       scale=alt.Scale(
                           domain=['5 panneaux', '10 panneaux', '20 panneaux', '50 panneaux'],
                           range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                       ),
                       legend=alt.Legend(title='Nombre de panneaux'))
    ).properties(
        width=800,
        height=400,
        title='Puissance nette en été - Différentes configurations PV'
    )

    # Add horizontal line at y=0 to show when production exceeds consumption
    _zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='gray',
        strokeDash=[5, 5],
        strokeWidth=1.5
    ).encode(y='y:Q')

    # Combine chart with zero line, then apply configuration
    _combined_chart = (_net_chart + _zero_line).configure_axis(
        labelFontSize=plot_config.font_size,
        titleFontSize=plot_config.font_size,
        grid=True,
        gridOpacity=0.3
    ).configure_title(
        fontSize=plot_config.font_size + 2
    ).configure_legend(
        labelFontSize=plot_config.font_size
    )

    # Create interactive marimo chart
    net_chart_ete = mo.ui.altair_chart(_combined_chart)
    net_chart_ete
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Puissance nette en hiver
    Visualisation de la puissance nette ( - production solai- consommation) en hiver pour différents nombres de panneaux PV
    """)
    return


@app.cell(hide_code=True)
def _(
    alt,
    charge,
    create_time_axis_encoding,
    mo,
    pd,
    plot_config,
    pv_config,
    solaire_hiver,
):

    # Extract winter consumption
    consumption_hiver = charge['Hiver'].values

    # Get the first numeric column from solar data (production values)
    solar_numeric_cols_hiver = solaire_hiver.select_dtypes(include=['float64', 'int64']).columns
    if len(solar_numeric_cols_hiver) > 0:
        # Use the first numeric column as production data
        production_reference_hiver = solaire_hiver[solar_numeric_cols_hiver[0]].values
    else:
        # Fallback if no numeric columns found
        production_reference_hiver = solaire_hiver.iloc[:, 0].values

    # Ensure both arrays have the same length
    min_length_hiver = min(len(consumption_hiver), len(production_reference_hiver))
    consumption_hiver = consumption_hiver[:min_length_hiver]
    production_reference_hiver = production_reference_hiver[:min_length_hiver]

    # Panel counts to analyze
    panel_counts_hiver = [5, 10, 20, 50]

    # Calculate net power for each panel count
    net_production_data_hiver = {'Index': range(min_length_hiver)}

    for _n_panels_hiver in panel_counts_hiver:
        # Scale production based on panel count
        _scaling_factor_hiver = _n_panels_hiver / pv_config.reference_panels
        production_scaled_hiver = production_reference_hiver * _scaling_factor_hiver

        # Calculate net power (production - consumption)
        net_production_hiver = production_scaled_hiver - consumption_hiver

        net_production_data_hiver[f'{_n_panels_hiver} panneaux'] = net_production_hiver

    # Create DataFrame in long format for Altair
    net_production_df_hiver = pd.DataFrame(net_production_data_hiver)
    net_production_long_hiver = net_production_df_hiver.melt(
        id_vars=['Index'],
        var_name='Configuration',
        value_name='Charge nette'
    )

    # Create Altair chart (without configure methods)
    _net_chart_hiver = alt.Chart(net_production_long_hiver).mark_line(
        strokeWidth=plot_config.line_width
    ).transform_calculate(
        ChargeNetteKW='datum["Charge nette"] / 1000'
    ).encode(
        x=create_time_axis_encoding(plot_config),
        y=alt.Y('ChargeNetteKW:Q', title='Puissance nette (kW)'),
        color=alt.Color('Configuration:N',
                       scale=alt.Scale(
                           domain=['5 panneaux', '10 panneaux', '20 panneaux', '50 panneaux'],
                           range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                       ),
                       legend=alt.Legend(title='Nombre de panneaux'))
    ).properties(
        width=800,
        height=400,
        title='Puissance nette en hiver - Différentes configurations PV'
    )

    # Add horizontal line at y=0 to show when production exceeds consumption
    _zero_line_hiver = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='gray',
        strokeDash=[5, 5],
        strokeWidth=1.5
    ).encode(y='y:Q')

    # Combine chart with zero line, then apply configuration
    _combined_chart_hiver = (_net_chart_hiver + _zero_line_hiver).configure_axis(
        labelFontSize=plot_config.font_size,
        titleFontSize=plot_config.font_size,
        grid=True,
        gridOpacity=0.3
    ).configure_title(
        fontSize=plot_config.font_size + 2
    ).configure_legend(
        labelFontSize=plot_config.font_size
    )

    # Create interactive marimo chart
    net_chart_hiver = mo.ui.altair_chart(_combined_chart_hiver)
    net_chart_hiver
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Autoproduction et autoconsommation
    Évolution des taux d'autoconsommation et d'autoproduction en fonction du nombre de panneaux PV
    """)
    return


@app.cell(hide_code=True)
def _(charge, np, pd, pv_config, solaire_ete, solaire_hiver):
    # Extract consumption data
    _consumption_ete = charge['Été'].values
    _consumption_hiver = charge['Hiver'].values

    # Extract production data (follow pattern from existing cells)
    _solar_ete_cols = solaire_ete.select_dtypes(include=['float64', 'int64']).columns
    if len(_solar_ete_cols) > 0:
        _production_ete_reference = solaire_ete[_solar_ete_cols[0]].values
    else:
        _production_ete_reference = solaire_ete.iloc[:, 0].values

    _solar_hiver_cols = solaire_hiver.select_dtypes(include=['float64', 'int64']).columns
    if len(_solar_hiver_cols) > 0:
        _production_hiver_reference = solaire_hiver[_solar_hiver_cols[0]].values
    else:
        _production_hiver_reference = solaire_hiver.iloc[:, 0].values

    # Ensure matching lengths
    _min_length = min(
        len(_consumption_ete),
        len(_consumption_hiver),
        len(_production_ete_reference),
        len(_production_hiver_reference)
    )
    _consumption_ete = _consumption_ete[:_min_length]
    _consumption_hiver = _consumption_hiver[:_min_length]
    _production_ete_reference = _production_ete_reference[:_min_length]
    _production_hiver_reference = _production_hiver_reference[:_min_length]

    # Initialize storage for results
    _autoconsommation_data = {'Nombre de panneaux': [], 'Été': [], 'Hiver': []}
    _autoproduction_data = {'Nombre de panneaux': [], 'Été': [], 'Hiver': []}

    # Calculate rates for each panel count (1 to 50)
    for _n_panels in range(1, 51):
        # Scale production from 21-panel reference
        _scaling_factor = _n_panels / pv_config.reference_panels
        _production_ete_scaled = _production_ete_reference * _scaling_factor
        _production_hiver_scaled = _production_hiver_reference * _scaling_factor

        # Calculate locally consumed energy (min at each timestep)
        _locally_consumed_ete = np.minimum(_production_ete_scaled, _consumption_ete)
        _locally_consumed_hiver = np.minimum(_production_hiver_scaled, _consumption_hiver)

        # Sum totals over all timesteps
        _total_consumption_ete = np.sum(_consumption_ete)
        _total_consumption_hiver = np.sum(_consumption_hiver)
        _total_production_ete = np.sum(_production_ete_scaled)
        _total_production_hiver = np.sum(_production_hiver_scaled)
        _total_locally_consumed_ete = np.sum(_locally_consumed_ete)
        _total_locally_consumed_hiver = np.sum(_locally_consumed_hiver)

        # Calculate autoconsommation rates (% of production consumed locally)
        _autoconso_ete = (_total_locally_consumed_ete / _total_production_ete) * 100 if _total_production_ete > 0 else 0.0
        _autoconso_hiver = (_total_locally_consumed_hiver / _total_production_hiver) * 100 if _total_production_hiver > 0 else 0.0

        # Calculate autoproduction rates (% of consumption produced locally)
        _autoprod_ete = (_total_locally_consumed_ete / _total_consumption_ete) * 100
        _autoprod_hiver = (_total_locally_consumed_hiver / _total_consumption_hiver) * 100

        # Store results
        _autoconsommation_data['Nombre de panneaux'].append(_n_panels)
        _autoconsommation_data['Été'].append(_autoconso_ete)
        _autoconsommation_data['Hiver'].append(_autoconso_hiver)
        _autoproduction_data['Nombre de panneaux'].append(_n_panels)
        _autoproduction_data['Été'].append(_autoprod_ete)
        _autoproduction_data['Hiver'].append(_autoprod_hiver)

    # Create DataFrames
    autoconsommation_df = pd.DataFrame(_autoconsommation_data)
    autoproduction_df = pd.DataFrame(_autoproduction_data)
    return autoconsommation_df, autoproduction_df


@app.cell(hide_code=True)
def _(
    alt,
    autoconsommation_df,
    autoproduction_df,
    mo,
    plot_config,
    rgb_to_hex,
):

    # Convert to long format for Altair
    autoconsommation_long = autoconsommation_df.melt(
        id_vars=['Nombre de panneaux'],
        var_name='Saison',
        value_name='Taux'
    )

    autoproduction_long = autoproduction_df.melt(
        id_vars=['Nombre de panneaux'],
        var_name='Saison',
        value_name='Taux'
    )

    # Create autoconsommation chart
    chart_autoconsommation = alt.Chart(autoconsommation_long).mark_line(
        strokeWidth=plot_config.line_width,
        point=True  # Add circular markers
    ).encode(
        x=alt.X('Nombre de panneaux:Q',
                title='Nombre de panneaux PV',
                scale=alt.Scale(domain=[1, 50])),
        y=alt.Y('Taux:Q',
                title='Taux d\'autoconsommation (%)',
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('Saison:N',
                       scale=alt.Scale(
                           domain=['Été', 'Hiver'],
                           range=[rgb_to_hex(plot_config.summer_color),
                                  rgb_to_hex(plot_config.winter_color)]
                       ),
                       legend=alt.Legend(title=None))
    ).properties(
        width=400,  # Half width for side-by-side
        height=400,
        title='Évolution du taux d\'autoconsommation'
    ).configure_axis(
        labelFontSize=plot_config.font_size,
        titleFontSize=plot_config.font_size,
        grid=True,
        gridOpacity=0.3
    ).configure_title(
        fontSize=plot_config.font_size + 2
    ).configure_legend(
        labelFontSize=plot_config.font_size
    )

    # Create autoproduction chart
    chart_autoproduction = alt.Chart(autoproduction_long).mark_line(
        strokeWidth=plot_config.line_width,
        point=True  # Add circular markers
    ).encode(
        x=alt.X('Nombre de panneaux:Q',
                title='Nombre de panneaux PV',
                scale=alt.Scale(domain=[1, 50])),
        y=alt.Y('Taux:Q',
                title='Taux d\'autoproduction (%)',
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('Saison:N',
                       scale=alt.Scale(
                           domain=['Été', 'Hiver'],
                           range=[rgb_to_hex(plot_config.summer_color),
                                  rgb_to_hex(plot_config.winter_color)]
                       ),
                       legend=alt.Legend(title=None))
    ).properties(
        width=400,  # Half width for side-by-side
        height=400,
        title='Évolution du taux d\'autoproduction'
    ).configure_axis(
        labelFontSize=plot_config.font_size,
        titleFontSize=plot_config.font_size,
        grid=True,
        gridOpacity=0.3
    ).configure_title(
        fontSize=plot_config.font_size + 2
    ).configure_legend(
        labelFontSize=plot_config.font_size
    )

    # Combine charts horizontally
    autoconso_chart = mo.ui.altair_chart(chart_autoconsommation)
    autoprod_chart = mo.ui.altair_chart(chart_autoproduction)
    autoprodconso_chart = mo.hstack([autoconso_chart, autoprod_chart])
    autoprodconso_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Critère composite (période estivale)
    Moyenne pondérée des taux d'autoconsommation et d'autoproduction.
    Ajustez le poids avec le curseur pour favoriser l'autoconsommation ou l'autoproduction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    weight_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.50,
        label="Poids autoconsommation",
        show_value=True
    )
    return (weight_slider,)


@app.cell(hide_code=True)
def compute_composite_criteria_pv_only(
    alt,
    autoconsommation_df,
    autoproduction_df,
    mo,
    pd,
    plot_config,
    weight_slider,
):

    # Get weight from slider
    _weight = weight_slider.value

    # Calculate composite criterion for summer (Été)
    _composite_ete = (
        _weight * autoconsommation_df['Été'] +
        (1 - _weight) * autoproduction_df['Été']
    )

    # Find maximum and corresponding panel count
    _max_composite = _composite_ete.max()
    _max_panel_count = autoconsommation_df.loc[_composite_ete.idxmax(), 'Nombre de panneaux']

    # Create DataFrame with all three curves
    _composite_df = autoconsommation_df[['Nombre de panneaux']].copy()
    _composite_df['Composite'] = _composite_ete
    _composite_df['Autoconsommation'] = autoconsommation_df['Été']
    _composite_df['Autoproduction'] = autoproduction_df['Été']

    # Convert to long format for Altair
    _composite_long = _composite_df.melt(
        id_vars=['Nombre de panneaux'],
        var_name='Critère',
        value_name='Valeur'
    )

    # Create color scheme
    _color_scale = alt.Scale(
        domain=['Composite', 'Autoconsommation', 'Autoproduction'],
        range=['#000000', '#FFA500', '#FF6347']  # Black, Orange, Tomato
    )

    # Create main chart with all three curves
    _chart_lines = alt.Chart(_composite_long).mark_line(
        strokeWidth=plot_config.line_width,
        point=True
    ).encode(
        x=alt.X('Nombre de panneaux:Q',
                title='Nombre de panneaux PV',
                scale=alt.Scale(domain=[1, 50])),
        y=alt.Y('Valeur:Q',
                title='Critère (%)',
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('Critère:N',
                       scale=_color_scale,
                       legend=alt.Legend(title='Légende'))
    )

    # Add red diamond marker at maximum
    _max_point = alt.Chart(
        pd.DataFrame({'x': [_max_panel_count], 'y': [_max_composite]})
    ).mark_point(
        size=200,
        color='red',
        shape='diamond',
        filled=True
    ).encode(
        x='x:Q',
        y='y:Q'
    )

    # Combine layers
    _combined_chart = (_chart_lines + _max_point).properties(
        width=800,
        height=400,
        title=f'Critère composite - Été (poids autoconso = {_weight:.2f}) -- Max = {_max_composite:.1f}% ({int(_max_panel_count)} panneaux)'
    ).configure_axis(
        labelFontSize=plot_config.font_size,
        titleFontSize=plot_config.font_size,
        grid=True,
        gridOpacity=0.3
    ).configure_title(
        fontSize=plot_config.font_size + 2
    ).configure_legend(
        labelFontSize=plot_config.font_size
    )

    # Wrap in marimo interactive chart
    _composite_chart = mo.ui.altair_chart(_combined_chart)
    _output = mo.vstack([
        weight_slider,
        _composite_chart
    ])
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dimensionnement de la batterie
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulation d'une configuration avec 50 panneaux et 30 kWh de batterie
    Pour cette première partie, nous allons modéliser un comportement simple d'une batterie de 30 kWh combinée à l'installation solaire de 50 panneaux.

    Les paramètres à visualiser sont:
    - La production PV
    - La consommation des bâtiments
    - La charge et décharge de la batterie
    - L'état de charge de la batterie
    - La charge sur le réseau
    """)
    return


@app.cell
def compute_values_simple_battery_model(
    battery_config,
    charge,
    mo,
    pd,
    pv_config,
    solaire_ete,
):
    # PARAMÈTRES CONFIGURABLES
    _n_panels = 50  # Fixe à 50 panneaux
    battery_capacity_wh_fixed = 30000  # 50 kWh par défaut

    # Extract summer consumption
    _consumption_ete = charge['Été'].values

    # Extract summer production data
    _solar_ete_cols = solaire_ete.select_dtypes(include=['float64', 'int64']).columns
    if len(_solar_ete_cols) > 0:
        _production_ete_reference = solaire_ete[_solar_ete_cols[0]].values
    else:
        _production_ete_reference = solaire_ete.iloc[:, 0].values

    # Ensure matching lengths
    _min_length = min(len(_consumption_ete), len(_production_ete_reference))
    _consumption_ete = _consumption_ete[:_min_length]
    _production_ete_reference = _production_ete_reference[:_min_length]

    # Scale production for 14 panels
    _scaling_factor = _n_panels / pv_config.reference_panels
    _production_scaled = _production_ete_reference * _scaling_factor

    # Battery simulation parameters
    _battery_efficiency = battery_config.efficiency
    _initial_soc = battery_config.initial_soc
    _min_soc = battery_config.min_soc
    _max_soc = battery_config.max_soc

    _initial_soe = _initial_soc * battery_capacity_wh_fixed * 60 # in W•minutes
    _current_soe = _initial_soe
    _battery_min_capacity = _min_soc * battery_capacity_wh_fixed * 60 # in W•minutes
    _battery_max_capacity = _max_soc * battery_capacity_wh_fixed * 60 # in W•minutes

    # Initialize timestep data storage
    _timestep_data = {
        'timestep': [],
        'current_soe_wh': [],
        'consumption_w': [],
        'production_w': [],
        'net_power_w' : [],
        'battery_power_w': [],  # Positive = charging, negative = discharging
        'grid_power_w': [],  # Positive = importing, negative = exporting
        'locally_consumed_w_minutes': [],
        'total_production_w_minutes': [],
        'total_consumption_w_minutes': [],
        'battery_soe_w_minutes': []
    }

    # Track energy flows for metric calculations
    _total_consumption = 0
    _total_production = 0
    _total_locally_consumed = 0

    # Timestep simulation
    for _i in range(len(_consumption_ete)):
        _consumption = _consumption_ete[_i]
        _production = _production_scaled[_i]
        _net_power = _production - _consumption

        # Update totals
        _total_consumption += _consumption  # Convert to W•minutes (assuming 1-minute timesteps)
        _total_production += _production # Convert to W•minutes

        if _net_power > 0:
        
            # Excess production
            _power_to_add = min(_net_power, _battery_max_capacity - _current_soe)
            _current_soe += _power_to_add
            _battery_power = _power_to_add # Charging
            _grid_power = _net_power - _power_to_add  # Excess to grid
            _locally_consumed = min(_production, _consumption + _power_to_add)
        
        else:
            # Deficit
            _power_needed = abs(_net_power)
            _discharge = min(_power_needed, _current_soe - _battery_min_capacity)
            _current_soe -= _discharge
            _battery_power = -_discharge  # Discharging
            _grid_power = -(_power_needed - _discharge)  # From grid
            _locally_consumed = min(_production, _consumption)

        # Store timestep data
        _timestep_data['timestep'].append(_i)
        _timestep_data['current_soe_wh'].append(_current_soe)
        _timestep_data['consumption_w'].append(_consumption_ete[_i])
        _timestep_data['production_w'].append(_production_scaled[_i])
        _timestep_data['net_power_w'].append(_net_power)
        _timestep_data['battery_power_w'].append(_battery_power)
        _timestep_data['grid_power_w'].append(_grid_power)
        _timestep_data['locally_consumed_w_minutes'].append(_locally_consumed)
        _timestep_data['total_production_w_minutes'].append(_total_production)
        _timestep_data['total_consumption_w_minutes'].append(_total_consumption)
        _timestep_data['battery_soe_w_minutes'].append(_current_soe)

    _total_locally_consumed = sum(_timestep_data['locally_consumed_w_minutes']) + (_current_soe - _initial_soe)
    _total_consumption = _timestep_data['total_consumption_w_minutes'][-1]
    _total_production = _timestep_data['total_production_w_minutes'][-1]

    timestep_df = pd.DataFrame(_timestep_data)
    mo.md(f"""Les données utilisées: 
    - Nombre de panneaux solaires: {_n_panels}
    - Taille de la batterie en kWh: {battery_capacity_wh_fixed / 1000}
    - Part autoconsommée: {_total_locally_consumed / _total_production * 100:.2f} %
    - Part autoproduite: {_total_locally_consumed / _total_consumption * 100:.2f} %

    {mo.ui.table(timestep_df)}""")
    return battery_capacity_wh_fixed, timestep_df


@app.cell(hide_code=True)
def visualize_simple_battery_model(
    alt,
    battery_capacity_wh_fixed,
    create_time_axis_encoding,
    mo,
    pd,
    plot_config,
    timestep_df,
):
    # Prepare data for visualization
    viz_df = timestep_df.copy()

    # Rename 'timestep' to 'Index' to work with create_time_axis_encoding function
    viz_df["Index"] = viz_df["timestep"]

    # Convert SOE to SOC percentage - Fixed variable name
    viz_df["soc_percent"] = (viz_df["current_soe_wh"] / battery_capacity_wh_fixed) * 100

    # Convert all power values to kW for better readability
    viz_df["consumption_kw"] = viz_df["consumption_w"] / 1000
    viz_df["production_kw"] = viz_df["production_w"] / 1000
    viz_df["battery_power_kw"] = viz_df["battery_power_w"] / 1000
    viz_df["grid_power_kw"] = viz_df["grid_power_w"] / 1000
    viz_df["net_power_kw"] = viz_df["net_power_w"] / 1000

    # Create power data in long format for the main chart
    power_data = viz_df[
        ["Index", "net_power_kw", "battery_power_kw", "grid_power_kw"]
    ].melt(id_vars=["Index"], var_name="Type", value_name="Power_kW")

    # Define colors and labels
    power_labels = {
        "net_power_kw": "Nette",
        "battery_power_kw": "Batterie",
        "grid_power_kw": "Réseau",
    }

    power_colors = ["#d62728", "#2ca02c", "#1f77b4"]  # Red, Green, Blue

    # Map labels in the dataframe
    power_data["Type_Label"] = power_data["Type"].map(power_labels)

    # Create main power chart (left y-axis) using existing time encoding function
    power_chart = (
        alt.Chart(power_data)
        .mark_line(strokeWidth=2)
        .encode(
            x=create_time_axis_encoding(plot_config),
            y=alt.Y("Power_kW:Q", title="Puissance (kW)"),
            color=alt.Color(
                "Type_Label:N",
                scale=alt.Scale(range=power_colors),
                legend=alt.Legend(title="Puissance"),
            ),
        )
    )

    # Create SOC chart (right y-axis) using existing time encoding function
    soc_chart = (
        alt.Chart(viz_df)
        .mark_line(strokeWidth=2, color="black", strokeDash=[5, 5])
        .encode(
            x=create_time_axis_encoding(plot_config),
            y=alt.Y(
                "soc_percent:Q",
                title="État de charge (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
        )
        .properties(width=1000, height=200, title="État de charge")
        .configure_axis(
            labelFontSize=plot_config.font_size,
            titleFontSize=plot_config.font_size,
            grid=True,
            gridOpacity=0.3,
        )
        .configure_title(fontSize=plot_config.font_size + 2)
        .configure_legend(labelFontSize=plot_config.font_size)
    )

    # Add horizontal line at y=0 for power reference
    zero_line = (
        alt.Chart(pd.DataFrame({"x": [0, viz_df["Index"].max()], "y": [0, 0]}))
        .mark_line(color="gray", strokeDash=[5, 5])
        .encode(x="x:Q", y="y:Q")
    )

    # Combine charts with independent y-axes
    final_chart = (
        alt.layer(power_chart + zero_line)
        .resolve_scale(y="independent")
        .properties(width=1000, height=500, title="Puissance")
        .configure_axis(
            labelFontSize=plot_config.font_size,
            titleFontSize=plot_config.font_size,
            grid=True,
            gridOpacity=0.3,
        )
        .configure_title(fontSize=plot_config.font_size + 2)
        .configure_legend(labelFontSize=plot_config.font_size)
    )

    # Create interactive marimo chart
    battery_chart = mo.ui.altair_chart(final_chart)
    soc_chart = mo.ui.altair_chart(soc_chart)
    mo.vstack([battery_chart, soc_chart])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulation des combinaisons de panneaux PV et de taille de batterie
    Dans cette section, nous nous occupons d'effectuer la simulation pour plusieurs tailles d'installations solaires et de batteries.
    """)
    return


@app.cell
def compute_composite_criteria_for_multiple_configurations(
    battery_config,
    charge,
    np,
    pv_config,
    solaire_ete,
):
    # Define parameter ranges
    pv_panel_range = range(0, 51, 1)  # 0 to 50 panels
    battery_capacity_range = range(0, 101, 5)  # 0 to 100 kWh in 5 kWh steps

    # Initialize results storage
    results = {
        'n_panels': [],
        'battery_capacity_kwh': [],
        'autoconsommation': [],
        'autoproduction': [],
        'composite_criterion': [],
        'total_locally_consumed_kWh': [],
        'total_consumption_kWh': [],
        'total_production_kWh': [],
        'initial_soe_kWh': [],
        'final_soe_kWh': []
    }

    # Extract summer consumption and production data (reuse from existing cells)
    _consumption_ete = charge['Été'].values
    _solar_ete_cols = solaire_ete.select_dtypes(include=['float64', 'int64']).columns
    if len(_solar_ete_cols) > 0:
        _production_ete_reference = solaire_ete[_solar_ete_cols[0]].values
    else:
        _production_ete_reference = solaire_ete.iloc[:, 0].values

    # Ensure matching lengths
    _min_length = min(len(_consumption_ete), len(_production_ete_reference))
    _consumption_ete = _consumption_ete[:_min_length]
    _production_ete_reference = _production_ete_reference[:_min_length]

    print(f"Running simulation for {len(pv_panel_range)} PV configurations × {len(battery_capacity_range)} battery sizes...")
    print(f"Total simulations: {len(pv_panel_range) * len(battery_capacity_range)}")

    # Run the comprehensive simulation
    for n_panels in pv_panel_range:
        for battery_capacity_kwh in battery_capacity_range:
            battery_capacity_w_minutes = battery_capacity_kwh * 1000 * 60 # Convert to W•minutes
        
            # Scale production for current panel count
            if n_panels == 0:
                _production_scaled = np.zeros_like(_production_ete_reference)
            else:
                _scaling_factor = n_panels / pv_config.reference_panels
                _production_scaled = _production_ete_reference * _scaling_factor
        
            # Battery simulation parameters
            _battery_efficiency = battery_config.efficiency
            _initial_soc = battery_config.initial_soc
            _min_soc = battery_config.min_soc
            _max_soc = battery_config.max_soc
        
            # Initialize battery state
            if battery_capacity_w_minutes == 0:
                # No battery case
                _initial_soe = 0
                _current_soe = 0
                _battery_min_capacity = 0
                _battery_max_capacity = 0
            else:
                _initial_soe = _initial_soc * battery_capacity_w_minutes # in W•minutes
                _current_soe = _initial_soe # in W•minutes
                _battery_min_capacity = _min_soc * battery_capacity_w_minutes # in W•minutes
                _battery_max_capacity = _max_soc * battery_capacity_w_minutes # in W•minutes
        
            # Track energy flows for metric calculations
            _total_consumption = 0 # in Wh
            _total_production = 0 # in Wh
            _total_locally_consumed = 0 # in Wh
        
            # Timestep simulation
            for _i in range(len(_consumption_ete)):
                _consumption = _consumption_ete[_i] # in W
                _production = _production_scaled[_i] # in W
                _net_power = _production - _consumption # in W
            
                # Update totals
                _total_consumption += _consumption  # in W•minutes (assuming 1-minute timesteps)
                _total_production += _production # in W•minutes
            
                if battery_capacity_w_minutes == 0:
                    # No battery - direct consumption
                    _locally_consumed = min(_production, _consumption) # in W•minutes
                else:
                    # Battery simulation
                    if _net_power > 0:
                        # Excess production
                        _power_to_add = min(_net_power, (_battery_max_capacity - _current_soe))
                        _current_soe += _power_to_add
                        _locally_consumed = min(_production, _consumption + _power_to_add)
                    else:
                        # Deficit
                        _power_needed = abs(_net_power)
                        _discharge = min(_power_needed, (_current_soe - _battery_min_capacity))
                        _current_soe -= _discharge
                        _locally_consumed = min(_production, _consumption)
                
                _total_locally_consumed += _locally_consumed
        
            # Calculate metrics
            _diff_soe = (_current_soe - _initial_soe)
            _total_locally_consumed += _diff_soe
        
            if _total_production > 0:
                _autoconsommation = (_total_locally_consumed / _total_production) * 100
            else:
                _autoconsommation = 100.0
        
            if _total_consumption > 0:
                _autoproduction = (_total_locally_consumed / _total_consumption) * 100
            else:
                _autoproduction = 0.0
        
            # Composite criterion (simple average)
            _composite = (_autoconsommation + _autoproduction) / 2
        
            # Store results
            results['n_panels'].append(n_panels)
            results['battery_capacity_kwh'].append(battery_capacity_kwh)
            results['autoconsommation'].append(_autoconsommation)
            results['autoproduction'].append(_autoproduction)
            results['composite_criterion'].append(_composite)
            results['total_locally_consumed_kWh'].append(_total_locally_consumed / 60 / 1000)
            results['total_consumption_kWh'].append(_total_consumption / 60 / 1000)
            results['total_production_kWh'].append(_total_production / 60 / 1000)
            results['initial_soe_kWh'].append(_initial_soe / 60 / 1000)
            results['final_soe_kWh'].append(_current_soe / 60 / 1000)

    print("Simulation completed!")
    return (results,)


@app.cell
def _(mo, results):
    mo.ui.table(results)
    return


@app.cell
def _(np, plot_config, plt, results):
    # Convert results to numpy arrays for 3D plotting
    pv_panels = np.array(results['n_panels'])
    battery_capacities = np.array(results['battery_capacity_kwh'])
    composite_values = np.array(results['composite_criterion'])

    # Create meshgrid for 3D surface plot
    PV_unique = np.unique(pv_panels)
    BAT_unique = np.unique(battery_capacities)
    PV_grid, BAT_grid = np.meshgrid(PV_unique, BAT_unique)

    # Reshape composite values to match meshgrid
    COMP_grid = composite_values.reshape(len(BAT_unique), len(PV_unique))

    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot
    surf = ax.plot_surface(PV_grid, BAT_grid, COMP_grid, 
                          cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)

    # Add contour lines on the bottom for better readability
    contours = ax.contour(PV_grid, BAT_grid, COMP_grid, zdir='z', 
                         offset=COMP_grid.min()-5, cmap='viridis', alpha=0.5)

    # Customize the plot
    ax.set_xlabel('Nombre de panneaux PV', fontsize=plot_config.font_size)
    ax.set_ylabel('Capacité batterie (kWh)', fontsize=plot_config.font_size)
    ax.set_zlabel('Critère composite (%)', fontsize=plot_config.font_size)
    ax.set_title('Optimisation PV + Batterie - Critère composite\n(Moyenne autoconsommation et autoproduction)', 
                 fontsize=plot_config.font_size + 2, pad=20)

    # Add color bar
    cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Critère composite (%)', fontsize=plot_config.font_size)

    # Find and mark the optimal point
    max_idx = np.argmax(composite_values)
    optimal_pv = results['n_panels'][max_idx]
    optimal_battery = results['battery_capacity_kwh'][max_idx]
    optimal_value = results['composite_criterion'][max_idx]

    ax.scatter([optimal_pv], [optimal_battery], [optimal_value], 
              color='red', s=100, alpha=1.0, marker='o')

    # Add text annotation for optimal point
    ax.text(optimal_pv, optimal_battery, optimal_value + 2, 
            f'Optimal:\n{optimal_pv} PV\n{optimal_battery} kWh\n{optimal_value:.1f}%',
            fontsize=plot_config.font_size - 1, ha='center')

    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
