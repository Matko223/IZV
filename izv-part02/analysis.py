from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile
import io
from matplotlib.patches import Patch

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz

# Ukol 1: nacteni dat ze ZIP souboru
def load_data(filename : str, ds : str) -> pd.DataFrame:
    """
    @brief Load data from a ZIP file
    @param filename: Path to the ZIP file
    @param ds: Dataset identifier
    @return: DataFrame containing the concatenated data from all years
    """
    
    years = ["2023", "2024", "2025"]
    tables = []

    # read zip file
    with zipfile.ZipFile(filename, 'r') as zf:
        files = zf.namelist()

        for year in years:
            zip_filename = f"{year}/I{ds}.xls"

            try:
                # read file from zip
                with zf.open(zip_filename) as file:
                    data_stream = io.BytesIO(file.read())
                    
                    dfs = pd.read_html(data_stream, encoding='cp1250', flavor='lxml')
                    df = dfs[0]

                    df.columns = [str(c).strip() for c in df.columns]
                    drop_cols = [c for c in df.columns if c == "" or c.lower().startswith("unnamed")]

                    if drop_cols:
                        df = df.drop(columns=drop_cols)

                    df = df.dropna(axis=1, how="all").copy()
                    
                    tables.append(df)

            except KeyError:
                print(f"Subor {zip_filename} neexistuje")
                continue

    result = pd.concat(tables, ignore_index=True)
    return result


# Ukol 2: zpracovani dat
def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    @brief Parse and clean the accident data
    @param df: Input DataFrame with raw accident data
    @param verbose: Print size information
    @return: Cleaned DataFrame with selected columns and transformations
    """

    region_map = {
        0: "PHA", 1: "STC", 2: "JHC", 3: "PLK", 4: "ULK", 5: "HKK",
        6: "JHM", 7: "MSK", 14: "OLK", 15: "ZLK", 16: "VYS",
        17: "PAK", 18: "LBK", 19: "KVK"
    }

    if df is None or df.empty:
        if verbose:
            print("new_size=0.0 MB")
        return pd.DataFrame()

    # check required columns
    required = ["p2a", "p4a", "p1"]
    if not all(col in df.columns for col in required):
        if verbose:
            print("new_size=0.0 MB")
        return pd.DataFrame()

    df_new = df.copy()

    df_new['date'] = pd.to_datetime(df_new['p2a'], errors='coerce')

    df_new['p4a'] = pd.to_numeric(df_new['p4a'], errors='coerce')
    df_new['region'] = df_new['p4a'].map(region_map)

    df_new = df_new.dropna(subset=['p1', 'date', 'region'])

    df_new = df_new.drop_duplicates(subset=['p1'], keep='first')
    df_new = df_new.reset_index(drop=True)

    if verbose:
        deep_size_bytes = df_new.memory_usage(deep=True).sum()
        new_size_mb = deep_size_bytes / 1_000_000
        print(f"new_size={new_size_mb:.1f} MB")

    return df_new

# Ukol 3: počty nehod v jednotlivých regionech podle stavu řidiče
def plot_state(df: pd.DataFrame, df_vehicles : pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    """
    @brief Plot number of accidents by driver state and region
    @param df: DataFrame with accident data
    @param df_vehicles: DataFrame with vehicle data
    @param fig_location: Path to save the figure
    @param show_figure: Whether to display the figure
    """

    states = {
        3: "pod vlivem léků, narkotik",
        4: "pod vlivem alkoholu, obsah alkoholu v krvi do 0,99 ‰",
        5: "pod vlivem alkoholu, obsah alkoholu v krvi 1 ‰ a více",
        6: "nemoc, úraz apod.",
        7: "invalida",
        8: "řidič při jízdě zemřel (infarkt apod.)",
        9: "pokus o sebevraždu, sebevražda",
    }
    
    # data
    accidents = df[['p1', 'region']].dropna(subset=['p1', 'region']).drop_duplicates(subset=['p1'])
    vehicles = df_vehicles[['p1', 'p57']].copy()
    vehicles['p57'] = pd.to_numeric(vehicles['p57'], errors='coerce')
    vehicles = vehicles[vehicles['p57'].isin(range(3, 10))]

    merged = vehicles.merge(accidents, on='p1', how='inner').dropna(subset=['region'])

    merged['state_label'] = (
        merged['p57']
        .astype(int)
        .map(states)
        .fillna(merged['p57'].astype(int).map(lambda x: f"Stav {x}"))
    )

    agg = merged.groupby(['state_label','region'])['p1'].nunique().unstack(fill_value=0)

    top_states = agg.sum(axis=1).sort_values(ascending=False).head(6).index.tolist()
    
    region_order = sorted(agg.columns.tolist())

    # graph
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes.flatten()

    base = sns.color_palette("tab10")
    palette = [base[i % len(base)] for i in range(len(region_order))]

    for i, state in enumerate(top_states):
        ax = axes[i]
        counts = agg.loc[state] if state in agg.index else pd.Series(0, index=agg.columns)
        counts = counts.reindex(region_order, fill_value=0)

        bar_data = pd.DataFrame({"region": counts.index, "count": counts.values})
        sns.barplot(data=bar_data, x="region", y="count", hue="region",
                    ax=ax, palette=palette, dodge=False, legend=False)

        ax.set_title(state, fontsize=12, fontweight='bold')
        ax.set_ylabel("Počet nehod", fontsize=10)
        ax.set_xlabel("Region", fontsize=10)
        ax.set_facecolor("#f7f9fb")

        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

        for p, v in zip(ax.patches, counts.values):
            ax.annotate(f"{int(v)}", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                        textcoords='offset points')

    fig.suptitle("Počet nehod podle stavu řidiče a kraje", fontsize=18, y=0.98)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)

    if fig_location:
        fig.savefig(fig_location, dpi=200, bbox_inches='tight')
    if show_figure:
        plt.show()
    plt.close(fig)

# Ukol4: alkohol a roky v krajích
def plot_alcohol(df: pd.DataFrame, df_consequences : pd.DataFrame, 
                 fig_location: str = None, show_figure: bool = False):
    """
    @brief Plot accidents under the influence of alcohol by consequences and region
    @param df: DataFrame with accident data
    @param df_consequences: DataFrame with consequences data
    @param fig_location: Path to save the figure
    @param show_figure: Display figure
    """

    if df.empty or df_consequences.empty:
        return

    required_cols = {'p1', 'p11', 'date', 'region'}
    if not required_cols.issubset(df.columns) or 'p1' not in df_consequences.columns or 'p59g' not in df_consequences.columns:
        return

    accidents = df[['p1', 'p11', 'date', 'region']].copy()
    accidents['p11'] = pd.to_numeric(accidents['p11'], errors='coerce')
    accidents = accidents.dropna(subset=['p1', 'p11', 'date', 'region'])

    merged = accidents.merge(df_consequences[['p1', 'p59g']], on='p1', how='inner')
    if merged.empty:
        return

    merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
    merged = merged.dropna(subset=['date'])
    merged['year'] = merged['date'].dt.year
    merged['month'] = merged['date'].dt.month

    alcohol = merged[(merged['p11'] >= 3) & (merged['month'] <= 10)].copy()
    if alcohol.empty:
        return

    alcohol['p59g'] = pd.to_numeric(alcohol['p59g'], errors='coerce')
    injury_map = {
        1: "Usmrcení",
        2: "Těžké zranění",
        3: "Lehké zranění",
        4: "Bez zranění",
    }
    alcohol['injury'] = alcohol['p59g'].map(injury_map)
    alcohol = alcohol.dropna(subset=['injury'])

    regions_alpha = sorted(alcohol['region'].dropna().unique().tolist())
    alcohol['region'] = pd.Categorical(alcohol['region'], categories=regions_alpha, ordered=True)

    agg = (alcohol
           .groupby(['injury', 'region', 'year'], dropna=False, observed=False)
           .size()
           .reset_index(name='count'))

    agg = agg[agg['region'].notna()]
    if agg.empty:
        return

    # graph
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=agg,
        kind="bar",
        col="injury",
        col_wrap=2,
        x="region",
        y="count",
        hue="year",
        palette="crest",
        sharey=False,
        height=5,
        aspect=1.4,
        legend=False
    )

    g.set_titles("Nasledky nehody: {col_name}")

    g.fig.set_size_inches(18, 12)

    agg['count'] = agg['count'].astype(int)

    for ax in g.axes.flatten():
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, None)

    # legend
    years = sorted(agg['year'].unique())
    palette = sns.color_palette("crest", n_colors=len(years))
    handles = [Patch(facecolor=palette[i], label=str(years[i])) for i in range(len(years))]
    g.fig.legend(handles=handles, title="Rok",
                 loc='lower right', bbox_to_anchor=(0.98, 0.02),
                 frameon=True, ncol=1)

    g.set_axis_labels("Kraj", "Počet nehod pod vlivem")
    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

    g.fig.suptitle("Nehody pod vlivem alkoholu podle následků (leden–říjen)", 
                   fontsize=16, fontweight="bold")
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.90, right=0.88, bottom=0.08)

    if fig_location:
        g.fig.savefig(fig_location, dpi=150, bbox_inches='tight')
    if show_figure:
        plt.show()
        
    plt.close(g.fig)

# Ukol 5: Podmínky v čase
def plot_conditions(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):
    """
    @brief Plot number of accidents by weather conditions over time
    @param df: DataFrame with accident data
    @param fig_location: Path to save the figure
    @param show_figure: Display figure
    """
    
    required = {'date', 'region', 'p18', 'p1'}
    if not required.issubset(df.columns):
        return

    selected_regions = ["PHA", "JHM", "MSK", "STC"]
    work = df[df['region'].isin(selected_regions)].copy()

    condition_map = {
        1: "Neztížené",
        2: "Mlha",
        3: "Počátek deště / slabý déšť",
        4: "Déšť",
        5: "Sněžení",
        6: "Námraza / náledí",
        7: "Nárazový vítr",
        0: "Jiné ztížené"
    }
    
    work['p18'] = pd.to_numeric(work['p18'], errors='coerce')
    work['condition'] = work['p18'].map(condition_map)
    work = work.dropna(subset=['condition', 'date'])
    
    work['date'] = pd.to_datetime(work['date'], errors='coerce')
    work = work.dropna(subset=['date'])
    
    monthly = (work
               .groupby(['region', pd.Grouper(key='date', freq='MS'), 'condition'])['p1']
               .nunique()
               .reset_index(name='count'))
    monthly_all = monthly.pivot_table(index=['region','date'], columns='condition', values='count', fill_value=0).reset_index()

    id_vars = ['date', 'region']
    value_vars = [c for c in monthly_all.columns if c not in id_vars]
    
    melted = monthly_all.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='Podmínky',
        value_name='count'
    )

    # filter
    melted['date'] = pd.to_datetime(melted['date'], errors='coerce')
    min_date_filter = pd.Timestamp('2023-01-01')
    max_date_filter = pd.Timestamp('2025-01-01')
    
    mask = (melted['date'] >= min_date_filter) & (melted['date'] <= max_date_filter)
    melted = melted[mask]

    # graph
    sns.set_theme(style="whitegrid")
    
    g = sns.relplot(
        data=melted,
        kind="line",
        x="date",
        y="count",
        hue="Podmínky",
        col="region",
        col_wrap=2,
        height=4.5,
        aspect=1.4,
        palette="tab10",
        facet_kws={'sharey': False},
        legend=False
    )

    g.set_titles("Kraj: {col_name}")
    g.set_axis_labels("", "Počet nehod")
    
    # x axis formatting
    max_date_plot = pd.Timestamp('2025-01-01') 
    
    for ax in g.axes.flatten():
        ax.set_xlim(min_date_filter, max_date_plot)
        
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.tick_params(axis='x', rotation=0)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('center')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # legend
    legend_labels = melted['Podmínky'].unique().tolist()
    palette_dict = dict(zip(legend_labels, sns.color_palette("tab10", len(legend_labels))))
    
    handles = [Patch(color=palette_dict[label], label=label) for label in legend_labels]
    
    g.fig.legend(handles=handles, labels=legend_labels, title="Povětrnostní podmínky",
                 loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=True)

    g.fig.suptitle("Počet nehod podle podmínek řidiče v čase (2023–2024)",
                   fontsize=16, fontweight="bold", y=0.98)
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.90, right=0.97)

    if fig_location:
        g.fig.savefig(fig_location, dpi=150, bbox_inches='tight')
    
    if show_figure:
        plt.show()
    
    plt.close(g.fig)

if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni
    # funkce.

    df = load_data("data_23_25.zip", "nehody")
    df_consequences = load_data("data_23_25.zip", "nasledky")
    df_vehicles = load_data("data_23_25.zip", "Vozidla")
    
    df2 = parse_data(df, True)
    
    plot_state(df2, df_vehicles, "01_state.png")
    plot_alcohol(df2, df_consequences, "02_alcohol.png", False)
    plot_conditions(df2, "03_conditions.png")

# Poznamka:
# pro to, abyste se vyhnuli castemu nacitani muzete vyuzit napr
# VS Code a oznaceni jako bunky (radek #%%% )
# Pak muzete data jednou nacist a dale ladit jednotlive funkce
# Pripadne si muzete vysledny dataframe ulozit nekam na disk (pro ladici
# ucely) a nacitat jej naparsovany z disku
