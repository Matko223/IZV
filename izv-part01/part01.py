#!/usr/bin/env python3
"""
IZV cast1 projektu 2025
Autor: Martin Valapka - xvalapm00

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene
na prednasce
"""
from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any
import re


def wave_inference_bad(
    x: NDArray[any], y: NDArray[any], sources: NDArray[any], wavelength: float
) -> NDArray[any]:
    """
    Referencni implementace, ktera je pomala a nevyuziva numpy efektivne;
    nezasahujte do ni!
    """
    k = 2 * np.pi / wavelength

    Z = np.zeros(x.shape + y.shape)
    for sx, sy in sources:
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                R = np.sqrt((x[i] - sx) ** 2 + (y[j] - sy) ** 2)
                Z[j, i] += np.cos(k * R) / (1 + R)
    return Z


def wave_inference(
    x: NDArray[any], y: NDArray[any], sources: NDArray[any], wavelength: float
) -> NDArray[any]:
    """
    @brief Computes wave interference pattern from multiple point sources using vectorized numpy operations.
    @param x 1D numpy array representing x-coordinates.
    @param y 1D numpy array representing y-coordinates.
    @param sources 2D numpy array of shape (n_sources, 2) containing source positions (x, y).
    @param wavelength The wavelength of the wave used to compute the wave number k.
    @return 2D numpy array Z representing the superposed wave field at each (x, y) point.
    """
    k = 2 * np.pi / wavelength

    # 2D grid of coordinates
    X, Y = np.meshgrid(x, y)

    # new 3rd dimension for broadcasting
    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]

    # Reshape sources for broadcasting
    sources = sources.reshape(1, 1, -1, 2)

    # Calculate x and y distances from grid points to all sources
    dx = X - sources[..., 0]
    dy = Y - sources[..., 1]

    R = np.sqrt(dx**2 + dy**2)

    Z = np.sum(np.cos(k * R) / (1 + R), axis=2)

    return Z


def plot_wave(
    Z: NDArray[any],
    x: NDArray[any],
    y: NDArray[any],
    show_figure: bool = False,
    save_path: str | None = None,
):
    """
    @brief Vykresli graf vlnoveho pole
    @param Z Wave field values as a 2D numpy array.
    @param x 1D numpy array representing x-coordinates.
    @param y 1D numpy array representing y-coordinates.
    @param show_figure If True, displays the plot.
    @param save_path If provided, saves the plot to the specified path.
    @return None
    """

    plt.figure(figsize=(10, 10))
    plt.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()),
               origin="lower", vmin=-1, vmax=1)
    plt.colorbar(label="Amplituda vlny")
    plt.xlabel("X pozicia")
    plt.ylabel("Y pozicia")
    plt.title("Vlnove pole")

    if save_path:
        plt.savefig(save_path)
    if show_figure:
        plt.show()
    plt.close()


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    @brief Generates a plot with sin and cos waves, highlighting areas between them and highlighting upper parts of each wave in secondary plot.
    @param show_figure If True, displays the plot.
    @param save_path If provided, saves the plot to the specified path.
    @return None
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    x = np.linspace(0, 4*np.pi, 1000)
    sin = np.sin(x)
    cos = np.cos(x)

    # --- First subplot ---
    ax1.fill_between(x, sin, cos, alpha=0.3, color='lightgreen')
    ax1.plot(x, sin, color='green', linewidth=1)
    ax1.plot(x, cos, color='green', linewidth=1)
    ax1.set_ylim(-1.5, 1.5)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylabel('f(x)')

    # --- Second subplot ---
    waves = np.array([sin, cos])
    wave_max = np.max(waves, axis=0)
    wave_min = np.min(waves, axis=0)

    # Create masks for coloring the maximum wave
    max_sin = np.ma.masked_where(sin < wave_max, wave_max)
    max_cos = np.ma.masked_where(cos < wave_max, wave_max)

    # Plot the colored maximum and dashed minimum
    ax2.plot(x, max_sin, color='blue', linestyle='-', linewidth=1)
    ax2.plot(x, max_cos, color='orange', linestyle='-', linewidth=1)
    ax2.plot(x, wave_min, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlim(0, 4*np.pi)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylabel('f(x)')

    # Custom LaTeX labels for fractions on the x-axis
    ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi,
             5*np.pi/2, 3*np.pi, 7*np.pi/2, 4*np.pi]
    labels = [r'$0$', r'$\dfrac{1}{2}\pi$', r'$\pi$', r'$\dfrac{3}{2}\pi$',
              r'$2\pi$', r'$\dfrac{5}{2}\pi$', r'$3\pi$', r'$\dfrac{7}{2}\pi$', r'$4\pi$']
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, fontsize=10, color='black')

    if save_path:
        plt.savefig(save_path)
    if show_figure:
        plt.show()
    plt.close(fig)


def download_data() -> Dict[str, List[Any]]:
    """
    @brief Downloads and parses station data from the specified URL.
    @return A dictionary containing lists of positions, latitudes, longitudes, and heights.
    """

    base = "https://ehw.fit.vutbr.cz/izv/"
    url = "https://ehw.fit.vutbr.cz/izv/stanice.html"
    html_text = requests.get(url).text

    # Extract the fragment name using regex
    loadstatic_match = re.search(r"loadstatic\(\s*'([^']+)'\s*,", html_text)

    fragment_name = loadstatic_match.group(1)
    fragment_resp = requests.get(f"{base}{fragment_name}")
    fragment_resp.encoding = 'utf-8'
    soup = BeautifulSoup(fragment_resp.text, 'html.parser')

    data = {
        "positions": [],
        "lats": [],
        "longs": [],
        "heights": []
    }

    # Parse the table rows
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        strong = cells[0].find("strong")

        # Skip rows without strong tag
        if not strong:
            continue

        # Extract data and convert to appropriate types
        position = strong.get_text(strip=True)
        lat = float(cells[2].get_text(strip=True).replace(",", ".")[:-1])
        long = float(cells[4].get_text(strip=True).replace(",", ".")[:-1])
        height = float(cells[6].get_text(strip=True).replace(
            "\xa0", "").replace(",", ".").strip())

        data["positions"].append(position)
        data["lats"].append(lat)
        data["longs"].append(long)
        data["heights"].append(height)

    return data


if __name__ == "__main__":
    X = np.linspace(-10, 10, 200)
    Y = np.linspace(-10, 10, 200)

    A = wave_inference_bad(X, Y, np.array([[-3, 0], [3, 0], [0, 4]]), 2)
    plot_wave(A, X, Y, show_figure=False)

    B = wave_inference(X, Y, np.array([[-3, 0], [3, 0], [0, 4]]), 2)
    plot_wave(B, X, Y, show_figure=False)

    generate_sinus(show_figure=True)

    data = download_data()
