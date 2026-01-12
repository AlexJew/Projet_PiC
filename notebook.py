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
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='gray',
        strokeDash=[5, 5],
        strokeWidth=1.5
    ).encode(y='y:Q')

    # Combine chart with zero line, then apply configuration
    combined_chart = (_net_chart + zero_line).configure_axis(
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
    net_chart_ete = mo.ui.altair_chart(combined_chart)
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
    zero_line_hiver = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='gray',
        strokeDash=[5, 5],
        strokeWidth=1.5
    ).encode(y='y:Q')

    # Combine chart with zero line, then apply configuration
    combined_chart_hiver = (_net_chart_hiver + zero_line_hiver).configure_axis(
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
    net_chart_hiver = mo.ui.altair_chart(combined_chart_hiver)
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

        # Calculate autoconsommation rates (% of consumption met by local production)
        _autoconso_ete = (_total_locally_consumed_ete / _total_consumption_ete) * 100
        _autoconso_hiver = (_total_locally_consumed_hiver / _total_consumption_hiver) * 100

        # Calculate autoproduction rates (% of production consumed locally)
        _autoprod_ete = (_total_locally_consumed_ete / _total_production_ete) * 100 if _total_production_ete > 0 else 0.0
        _autoprod_hiver = (_total_locally_consumed_hiver / _total_production_hiver) * 100 if _total_production_hiver > 0 else 0.0

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
def _(
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
    ### Algorithme estival

    La première partie se focalise sur l'identification de la plage de dimensionnement pour le système PV et le système de batterie sur la période estivale.
    """)
    return


@app.cell
def compute_battery_behavior(
    battery_config,
    charge,
    mo,
    np,
    pd,
    pv_config,
    solaire_ete,
):
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

    # Ensure matching lengths
    _min_length = min(len(_consumption_ete), len(_production_ete_reference))
    _consumption_ete = _consumption_ete[:_min_length]
    _production_ete_reference = _production_ete_reference[:_min_length]

    # Define ranges
    _pv_range = range(1, 51)
    _battery_range = np.arange(0, 105, 5)  * 1000 # 0 to 100 kWh, 5 kWh steps

    # Initialize storage for results
    _results_data = {
        'pv_panels': [],
        'battery_capacity_wh': [],
        'autoconso' : [],
        'autoprod' : [],
        'composite': []
    }

    # Battery parameters
    _battery_efficiency = battery_config.efficiency
    _initial_soc = battery_config.initial_soc
    _min_soc = battery_config.min_soc
    _max_soc = battery_config.max_soc

    # Loop through all combinations
    for _n_panels in _pv_range:
        # Scale production for this panel count
        _scaling_factor = _n_panels / pv_config.reference_panels
        _production_scaled = _production_ete_reference * _scaling_factor

        for _battery_capacity in _battery_range:
            # Skip if battery capacity is 0 - no battery simulation needed
            if _battery_capacity == 0:
                # Calculate without battery
                _locally_consumed = np.minimum(_production_scaled, _consumption_ete)
                _total_consumption = np.sum(_consumption_ete)
                _total_production = np.sum(_production_scaled)
                _total_locally_consumed = np.sum(_locally_consumed)

                _autoconso = (_total_locally_consumed / _total_consumption) * 100
                _autoprod = (_total_locally_consumed / _total_production) * 100 if _total_production > 0 else 0.0
                _composite = 0.5 * _autoconso + 0.5 * _autoprod

                _results_data['pv_panels'].append(_n_panels)
                _results_data['battery_capacity_wh'].append(_battery_capacity)
                _results_data['autoconso'].append(_autoconso)
                _results_data['autoprod'].append(_autoprod)
                _results_data['composite'].append(_composite)
                continue

            # Battery simulation
            _initial_soe = _initial_soc * _battery_capacity
            _current_soe = _initial_soe
            _battery_min_capacity = _min_soc * _battery_capacity
            _battery_max_capacity = _max_soc * _battery_capacity

            _energy_from_grid = 0.0
            _excess_sold = 0.0

            # Timestep simulation
            for _i in range(len(_consumption_ete)):
                _net_power = _production_scaled[_i] - _consumption_ete[_i]

                if _net_power > 0:
                    # Excess production
                    _charge_to_add = min(_net_power, _battery_max_capacity - _current_soe)
                    _current_soe += _charge_to_add * _battery_efficiency
                    _excess_sold += _net_power - _charge_to_add
                else:
                    # Deficit
                    _energy_needed = abs(_net_power)
                    _discharge = min(_energy_needed / _battery_efficiency, _current_soe - _battery_min_capacity)
                    _current_soe -= _discharge
                    _energy_from_grid += _energy_needed - _discharge * _battery_efficiency

            # Correct for SOE difference
            _final_soe_diff = _current_soe - _initial_soe

            # Calculate metrics
            _total_consumption = np.sum(_consumption_ete)
            _total_production = np.sum(_production_scaled)

            _autoconso = ((_total_consumption - _energy_from_grid) / _total_consumption) * 100

            if _final_soe_diff > 0:
                # Battery gained energy - subtract from autoproduction
                _autoprod = ((_total_production - _excess_sold - _final_soe_diff) / _total_production) * 100
            else:
                # Battery lost energy - already accounted in autoconsumption
                _autoprod = ((_total_production - _excess_sold) / _total_production) * 100

            _composite = 0.5 * _autoconso + 0.5 * _autoprod

            _results_data['pv_panels'].append(_n_panels)
            _results_data['battery_capacity_wh'].append(_battery_capacity)
            _results_data['autoconso'].append(_autoconso)
            _results_data['autoprod'].append(_autoprod)
            _results_data['composite'].append(_composite)

    # Create DataFrame
    results_3d_df = pd.DataFrame(_results_data)
    mo.ui.table(_results_data)
    return (results_3d_df,)


@app.cell
def _(results_3d_df):
    results_3d_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analyse temporelle du comportement batterie

    Visualisation de l'état d'énergie (SOE), de la production PV et de la consommation
    pour une configuration fixe (14 panneaux, capacité batterie configurable).
    """)
    return


@app.cell(hide_code=True)
def simulate_timestep_analysis(battery_config, charge, mo, np, pd, pv_config, solaire_ete):
    # PARAMÈTRES CONFIGURABLES
    _n_panels = 14  # Fixe à 14 panneaux
    _battery_capacity_wh = 50000  # 50 kWh par défaut

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

    _initial_soe = _initial_soc * _battery_capacity_wh
    _current_soe = _initial_soe
    _battery_min_capacity = _min_soc * _battery_capacity_wh
    _battery_max_capacity = _max_soc * _battery_capacity_wh

    # Initialize timestep data storage
    _timestep_data = {
        'timestep': [],
        'current_soe_wh': [],
        'consumption_w': [],
        'production_w': [],
        'battery_power_w': [],  # Positive = charging, negative = discharging
        'grid_power_w': []  # Positive = importing, negative = exporting
    }

    # Timestep simulation
    for _i in range(len(_consumption_ete)):
        _net_power = _production_scaled[_i] - _consumption_ete[_i]

        if _net_power > 0:
            # Excess production
            _charge_to_add = min(_net_power, _battery_max_capacity - _current_soe)
            _current_soe += _charge_to_add * _battery_efficiency
            _battery_power = _charge_to_add  # Charging
            _grid_power = _net_power - _charge_to_add  # Excess to grid
        else:
            # Deficit
            _energy_needed = abs(_net_power)
            _discharge = min(_energy_needed / _battery_efficiency, _current_soe - _battery_min_capacity)
            _current_soe -= _discharge
            _battery_power = -_discharge * _battery_efficiency  # Discharging
            _grid_power = -(_energy_needed - _discharge * _battery_efficiency)  # From grid

        # Store timestep data
        _timestep_data['timestep'].append(_i)
        _timestep_data['current_soe_wh'].append(_current_soe)
        _timestep_data['consumption_w'].append(_consumption_ete[_i])
        _timestep_data['production_w'].append(_production_scaled[_i])
        _timestep_data['battery_power_w'].append(_battery_power)
        _timestep_data['grid_power_w'].append(_grid_power)

    timestep_df = pd.DataFrame(_timestep_data)

    return _battery_capacity_wh, _n_panels, timestep_df


@app.cell(hide_code=True)
def plot_timestep_analysis(_battery_capacity_wh, _n_panels, mo, np, plot_config, plt, timestep_df):
    # Prepare data (convert minutes to hours)
    _time_hours = timestep_df['timestep'].values / 60

    # Create figure with subplots
    _fig, _axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Production and Consumption
    _axes[0].plot(_time_hours, timestep_df['production_w'].values / 1000,
                  label='Production PV', color='orange', linewidth=plot_config.line_width)
    _axes[0].plot(_time_hours, timestep_df['consumption_w'].values / 1000,
                  label='Consommation', color='blue', linewidth=plot_config.line_width)
    _axes[0].set_ylabel('Puissance (kW)', fontsize=plot_config.font_size)
    _axes[0].legend(fontsize=plot_config.font_size)
    _axes[0].grid(True, alpha=0.3)
    _axes[0].set_title('Production PV et Consommation', fontsize=plot_config.font_size + 1)

    # Plot 2: Battery SOE
    _axes[1].plot(_time_hours, timestep_df['current_soe_wh'].values / 1000,
                  label='SOE batterie', color='green', linewidth=plot_config.line_width)
    _axes[1].set_ylabel('Énergie (kWh)', fontsize=plot_config.font_size)
    _axes[1].legend(fontsize=plot_config.font_size)
    _axes[1].grid(True, alpha=0.3)
    _axes[1].set_title('État d\'énergie de la batterie', fontsize=plot_config.font_size + 1)

    # Plot 3: Battery Power and Grid Power
    _axes[2].plot(_time_hours, timestep_df['battery_power_w'].values / 1000,
                  label='Puissance batterie', color='green', linewidth=plot_config.line_width)
    _axes[2].plot(_time_hours, timestep_df['grid_power_w'].values / 1000,
                  label='Puissance réseau', color='red', linewidth=plot_config.line_width, alpha=0.7)
    _axes[2].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    _axes[2].set_xlabel('Temps (heures)', fontsize=plot_config.font_size)
    _axes[2].set_ylabel('Puissance (kW)', fontsize=plot_config.font_size)
    _axes[2].legend(fontsize=plot_config.font_size)
    _axes[2].grid(True, alpha=0.3)
    _axes[2].set_title('Puissance batterie et réseau', fontsize=plot_config.font_size + 1)

    plt.tight_layout()

    # Display configuration info
    mo.md(f"""
    **Configuration analysée:**
    - Panneaux PV: {_n_panels}
    - Capacité batterie: {_battery_capacity_wh / 1000:.0f} kWh
    """)
    return


@app.cell
def _(mo, np, plot_config, plt, results_3d_df):
    from mpl_toolkits.mplot3d import Axes3D

    # Prepare data for 3D plot
    _pv_range = results_3d_df['pv_panels'].unique()
    _battery_range = results_3d_df['battery_capacity_wh'].unique()
    _X, _Y = np.meshgrid(_pv_range, _battery_range)
    _Z = results_3d_df['composite'].values.reshape(len(_battery_range), len(_pv_range))

    # Create 3D figure
    _fig = plt.figure(figsize=(12, 8))
    _ax = _fig.add_subplot(111, projection='3d')

    # Plot surface
    _surf = _ax.plot_surface(_X, _Y, _Z, cmap='viridis', edgecolor='none', alpha=0.9)

    # Add labels and title
    _ax.set_xlabel('Nombre de panneaux PV', fontsize=plot_config.font_size)
    _ax.set_ylabel('Capacité batterie (Wh)', fontsize=plot_config.font_size)
    _ax.set_zlabel('Critère composite (%)', fontsize=plot_config.font_size)
    _ax.set_title('Critère composite - Système PV + Batterie (Été)', fontsize=plot_config.font_size + 2)

    # Add colorbar
    _fig.colorbar(_surf, ax=_ax, shrink=0.5, aspect=5, label='Critère composite (%)')

    # Adjust viewing angle
    _ax.view_init(elev=25, azim=45)

    plt.tight_layout()

    # Find and mark maximum
    _max_idx = results_3d_df['composite'].idxmax()
    _max_pv = results_3d_df.loc[_max_idx, 'pv_panels']
    _max_battery = results_3d_df.loc[_max_idx, 'battery_capacity_wh']
    _max_composite = results_3d_df.loc[_max_idx, 'composite']

    # Display statistics
    mo.md(f"""
    **Optimum global:**
    - Panneaux PV: {int(_max_pv)}
    - Capacité batterie: {_max_battery:.0f} Wh ({_max_battery / 1000:.0f} kWh)
    - Critère composite: {_max_composite:.2f}%
    """)
    return


if __name__ == "__main__":
    app.run()
