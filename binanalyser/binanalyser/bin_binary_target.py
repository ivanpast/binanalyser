from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
from optbinning import OptimalBinning
from scipy import stats
from scipy.stats import somersd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_excel(df, filename="binning_results.xlsx"):
    """Genera un archivo Excel con los datos de la muestra y los tramos asignados."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Binning Results')
    processed_data = output.getvalue()
    return processed_data

def download_button(data, filename, label):
    """Crea un botón para descargar los datos como un archivo Excel."""
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def interactive_binning(df):
    st.title("Interactive Binning")

    # Selección de variables numéricas
    numeric_columns = df.select_dtypes(exclude=['object', 'category','datetime64[ns]', 'datetime64']).columns.tolist()
    if not numeric_columns:
        st.error("No numeric columns found in the dataset.")
        return

    continuous_var = st.selectbox("Select numerical variable for binning", numeric_columns)
    target_var = st.selectbox("Select target variable", df.columns)

    # Configuración del binning
    max_n_bins = st.slider("Maximum number of bins", 2, 20, 10)
    min_bin_size = st.slider("Minimum bin size (%)", 0.0, 20.0, 5.0) / 100
    monotonic_trend = st.selectbox("Monotonic trend", ["auto", "ascending", "descending", "convex", "concave", "none"])

    # Realizar binning inicial
    optb = OptimalBinning(name=continuous_var,
                          dtype="numerical",
                          solver="cp",
                          max_n_bins=max_n_bins,
                          min_bin_size=min_bin_size,
                          monotonic_trend=monotonic_trend)

    optb.fit(df[continuous_var], df[target_var])

    # Validar y limpiar los splits
    valid_splits = validate_splits(optb.splits)

    # Calcular HHI y Somers' D para los splits automáticos
    table = optb.binning_table.build()
    table = table[~table['Bin'].isin(['Special', 'Missing'])]
    table['Bin'] = table['Bin'].apply(lambda x: str(x))
    hhi_value = calculate_herfindahl(table)

    if valid_splits is not None and len(valid_splits) > 0:
        somers_d_value = calculate_somersd(df, continuous_var, target_var, valid_splits)
    else:
        somers_d_value = np.nan
        st.warning("No valid splits found. Somers' D could not be calculated.")

    # Mostrar la tabla con los bins, HHI y Somers' D automáticos
    st.subheader("Binning Results (Automatic)")
    st.write(table)

    # Botón para descargar la muestra completa con tramos asignados
    excel_data = generate_excel(df)
    download_button(excel_data, "binning_results.xlsx", "Download full sample with bins")

    st.subheader("HHI and Somers' D (Automatic)")
    hhi_value = calculate_herfindahl(table)["HI"]
    st.write(f"Herfindahl-Hirschman Index (HHI): {hhi_value:.4f}")
    st.write(f"Somers' D: {somers_d_value:.4f}")

    # Visualización de violin plots con p-valores para los splits automáticos
    if valid_splits is not None and len(valid_splits) > 0:
        df['bin'] = pd.cut(df[continuous_var], bins=valid_splits, include_lowest=True)
        plot_violinplots_with_pvalues(df, continuous_var, target_var)

    # Selección y combinación de splits manual
    st.subheader("Manual Split Combination")
    st.write(f"Initial Splits: {valid_splits[1:-1]}")  # Mostrar los splits sin los infinitos

    if valid_splits is not None and len(valid_splits) > 0:
        selected_splits = st.multiselect("Select splits to combine", options=valid_splits[1:-1],
                                         default=valid_splits[1:-1])
        if st.button("Combine Selected Splits"):
            combined_splits = combine_splits(valid_splits, selected_splits)
            st.write(f"New Combined Splits: {combined_splits[1:-1]}")

            # Aplicar los splits combinados a los datos
            df['bin'] = pd.cut(df[continuous_var], bins=combined_splits, include_lowest=True)

            # Calcular y mostrar la nueva tabla de resultados
            new_table = calculate_metrics(df, continuous_var, target_var, combined_splits)
            st.subheader("Updated Binning Results (Manual)")
            st.write(new_table)

            # Visualizar la distribución con los nuevos splits
            plot_violinplots_with_pvalues(df, continuous_var, target_var)

            # Botón para descargar la muestra actualizada con los nuevos tramos asignados
            excel_data = generate_excel(df)
            download_button(excel_data, "updated_binning_results.xlsx", "Download updated sample with bins")


def plot_violinplots_with_pvalues(df, continuous_var, target_var):
    bins = df['bin'].cat.categories
    p_values = []

    for i in range(len(bins) - 1):
        bin1_data = df[df['bin'] == bins[i]][target_var]
        bin2_data = df[df['bin'] == bins[i + 1]][target_var]
        if len(bin1_data) > 0 and len(bin2_data) > 0:
            _, p_value = stats.ranksums(bin1_data, bin2_data)
            p_values.append(p_value)
        else:
            p_values.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 4))  # Cambia estos valores para ajustar el tamaño del gráfico
    sns.violinplot(x='bin', y=continuous_var, data=df, ax=ax, inner='quartile', palette='muted')

    # Añadir p-valores en la gráfica
    for i, p_value in enumerate(p_values):
        x1, x2 = i, i + 1
        y, h, col = df[continuous_var].max() + df[continuous_var].std() / 2, df[continuous_var].std() / 10, 'k'
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
        formatted_p_value = f"p={p_value:.4f}"  # Redondear a 4 decimales
        ax.text((x1 + x2) * 0.5, y + h, formatted_p_value, ha='center', va='bottom', color=col, fontsize=10)

    ax.set_title(f"Violin Plots of {continuous_var} with Wilcoxon p-values", fontsize=12)
    ax.set_xlabel("Bins", fontsize=10)
    ax.set_ylabel(continuous_var, fontsize=10)
    sns.despine()
    st.pyplot(fig)


def validate_splits(splits):
    """ Valida y limpia los splits eliminando duplicados y NaN """
    if splits is None:
        return []
    splits = np.unique([split for split in splits if not pd.isna(split)])
    if len(splits) < 2:
        return []
    return [-np.inf] + splits.tolist() + [np.inf]


def combine_splits(splits, selected_splits):
    """ Combina todos los splits, excepto los seleccionados """
    new_splits = [-np.inf]  # Always include the lower bound

    for split in splits[1:-1]:  # Ignore the first and last (infinites)
        if split in selected_splits:
            new_splits.append(split)

    new_splits.append(np.inf)  # Always include the upper bound
    return np.unique(new_splits)


def calculate_metrics(df, continuous_var, target_var, splits):
    """Calcula HHI, Somers' D y devuelve una tabla con los resultados, incluyendo etiquetas para los nuevos bins."""
    # Create labeled bins
    df['bin'] = pd.cut(df[continuous_var], bins=splits, include_lowest=True)
    labels = [f"({bin.left:.2f}, {bin.right:.2f}]" for bin in df['bin'].cat.categories]
    df['bin_label'] = pd.cut(df[continuous_var], bins=splits, include_lowest=True, labels=labels)

    # Group by the labeled bins
    table = df.groupby('bin_label').agg(
        Count=('bin_label', 'size'),
        Non_event=(target_var, lambda x: (x == 0).sum()),
        Event=(target_var, lambda x: (x == 1).sum())
    )
    table['Event rate'] = table['Event'] / table['Count']
    table['WoE'] = np.log((table['Event'] / table['Event'].sum()) / (table['Non_event'] / table['Non_event'].sum()))
    table['IV'] = (table['Event'] / table['Event'].sum() - table['Non_event'] / table['Non_event'].sum()) * table['WoE']
    # Calcular el HHI usando la nueva función calculate_herfindahl
    hhi_result = calculate_herfindahl(table)
    table['HHI'] = hhi_result["HI"]  # Use HI_trad or HI as required
    #table['HHI'] = calculate_herfindahl(table)
    somers_d_value = calculate_somersd(df, continuous_var, target_var, splits)
    table['SomersD'] = somers_d_value
    table['IV'] = table['IV'].sum()

    # Reset the index to display the bin labels clearly
    table = table.reset_index()

    return table


def calculate_herfindahl(table):
    counts = table['Count']
    total = counts.sum()

    # Calcular las frecuencias
    concentration = counts.value_counts().reset_index()
    concentration.columns = ['HRC', 'Nobs']
    concentration['freq'] = concentration['Nobs'] / total

    # Número de categorías (K)
    K = len(concentration)
    coef = 1 / K

    # Calcular el índice
    concentration['index'] = (concentration['freq'] - coef) ** 2

    # Calcular CV
    CV2 = (np.sqrt(np.sum(concentration['index']) * K)) ** 2
    CV = np.sqrt(np.sum(concentration['index']) * K)

    # Calcular HI
    HI = 1 + np.log((CV2 + 1) / K) / np.log(K)

    # Calcular el HI tradicional
    HI_trad = np.sum(concentration['freq'] ** 2)

    return {
        "HI": HI,
        "CV": CV,
        "HI_trad": HI_trad,
        "table": concentration
    }


def calculate_somersd(df, continuous_var, target_var, splits):
    df['bin'] = pd.cut(df[continuous_var], bins=splits, include_lowest=True)
    df['bin_code'] = pd.Categorical(df['bin']).codes
    somers_d_result = somersd(df['bin_code'], df[target_var])
    return somers_d_result.statistic  # Acceder al valor de Somers' D


def main():
    st.set_page_config(layout="wide")

    # File uploader that accepts multiple file formats
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "sas7bdat"])

    if uploaded_file is not None:
        # Detect file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.sas7bdat'):
            df = pd.read_sas(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return

        interactive_binning(df)
    else:
        st.write("Please upload a CSV, Excel, or SAS file to begin.")


if __name__ == "__main__":
    main()
