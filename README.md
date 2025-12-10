**Dokumentacja Techniczna Projektu: System Predykcji Cen na Giełdach**

**Przedmiot:** Metody Sztucznej Inteligencji

**Temat:** Zastosowanie architektury hybrydowej (Autoenkoder + LSTM) w
detekcji anomalii i prognozowaniu notowań giełdowych.

**Technologia:** Python, TensorFlow (Keras), Streamlit

Autorzy: Natalia Klimaszewska, Roksana Żyłka

**1. Cel i Zakres Projektu**

Celem projektu jest opracowanie i implementacja systemu informatycznego
służącego do prognozowania wartości instrumentów finansowych w oparciu o
metody Deep Learning.

Głównym problemem badawczym jest wrażliwość klasycznych modeli
predykcyjnych na szum informacyjny i anomalie rynkowe (tzw. \"czarne
łabędzie\", nagłe krachy). W celu mityzacji tego problemu zastosowano
podejście hybrydowe, składające się z dwóch modułów:

1.  **Moduł nienadzorowany (Autoenkoder):** Odpowiedzialny za ekstrakcję
    cech i filtrację anomalii.

2.  **Moduł nadzorowany (LSTM):** Odpowiedzialny za właściwą predykcję
    na podstawie oczyszczonych danych.

**2. Podstawy Teoretyczne**

**2.1. Autoenkoder (Sieć Autoasocjacyjna)**

W projekcie wykorzystano symetryczną sieć typu *Feed-Forward* z warstwą
ukrytą o zredukowanej liczbie neuronów (tzw. *bottleneck*). Celem sieci
jest aproksymacja funkcji tożsamościowej.

- **Zasada działania:** Poprzez wymuszenie kompresji danych w warstwie
  ukrytej, sieć uczy się generalizacji wzorców typowych dla danego
  zbioru danych.

- **Detekcja anomalii:** Dane nietypowe (anomalie) charakteryzują się
  inną strukturą statystyczną niż dane treningowe. Podczas próby ich
  rekonstrukcji sieć generuje wysoki błąd odwzorowania (Mean Squared
  Error - MSE). Wartość błędu rekonstrukcji służy jako metryka decyzyjna
  do klasyfikacji punktu jako anomalii.

**2.2. Sieć LSTM (Long Short-Term Memory)**

Do prognozowania szeregu czasowego wykorzystano rekurencyjną sieć
neuronową (RNN) typu LSTM. Architektura ta rozwiązuje problem
zanikającego gradientu występujący w klasycznych sieciach RNN, co
pozwala na efektywne uczenie się długoterminowych zależności w danych
(tzw. *long-term dependencies*).

**3. Architektura Systemu**

System realizuje przetwarzanie potokowe składające się z następujących
etapów:

1.  **Akwizycja Danych:** Pobranie historycznych danych OHLC (Open,
    High, Low, Close) z serwisu Yahoo Finance.

2.  **Preprocessing:**

    - Ekstrakcja ceny zamknięcia (Close).

    - Normalizacja Min-Max do przedziału \$\[0, 1\]\$.

    - Okienkowanie (Sliding Window) -- transformacja szeregu do postaci
      sekwencji o długości \$N=40\$.

3.  **Filtracja (Autoenkoder):**

    - Trening modelu na surowych danych.

    - Wyznaczenie progu odcięcia na podstawie 95. centyla błędu
      rekonstrukcji (MAE).

    - Identyfikacja i imputacja anomalii (zastąpienie wartością z kroku
      \$t-1\$).

4.  **Predykcja (LSTM):**

    - Trening modelu na danych oczyszczonych.

    - Generowanie prognozy iteracyjnej (autoregresyjnej) na zadaną
      liczbę dni.

**4. Specyfikacja Implementacyjna Modeli**

**4.1. Model 1: Autoenkoder (Detektor Anomalii)**

<table>
<colgroup>
<col style="width: 27%" />
<col style="width: 72%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>Parametr</strong></th>
<th><strong>Wartość / Opis</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><strong>Typ sieci</strong></td>
<td>Dense (Pełna połączeniowa), Symetryczna</td>
</tr>
<tr class="even">
<td><strong>Warstwa wejściowa</strong></td>
<td>40 neuronów (odpowiada WINDOW_SIZE)</td>
</tr>
<tr class="odd">
<td><strong>Enkoder</strong></td>
<td>Warstwa Dense (32 neurony, aktywacja ReLU)<br />
Warstwa Bottleneck (16 neuronów, aktywacja ReLU)</td>
</tr>
<tr class="even">
<td><strong>Dekoder</strong></td>
<td>Warstwa Dense (32 neurony, aktywacja ReLU)<br />
Warstwa Wyjściowa (40 neuronów, aktywacja Sigmoid)</td>
</tr>
<tr class="odd">
<td><strong>Funkcja Straty</strong></td>
<td>Mean Squared Error (MSE)</td>
</tr>
<tr class="even">
<td><strong>Liczba epok</strong></td>
<td>40</td>
</tr>
</tbody>
</table>

Na wyjściu zastosowano funkcję Sigmoid, ponieważ dane wejściowe są
znormalizowane do przedziału \[0, 1\]. Funkcja ReLU w warstwach ukrytych
zapobiega problemowi zanikającego gradientu i przyspiesza konwergencję.

**4.2. Model 2: LSTM (Predyktor)**

| **Parametr**          | **Wartość / Opis**                                   |
|-----------------------|------------------------------------------------------|
| **Typ sieci**         | Recurrent Neural Network (LSTM)                      |
| **Kształt wejścia**   | (Batch_Size, 40, 1)                                  |
| **Warstwy ukryte**    | 1 warstwa LSTM, 50 jednostek (units), aktywacja ReLU |
| **Warstwa wyjściowa** | Dense (1 neuron) -- regresja liniowa                 |
| **Funkcja Straty**    | Mean Squared Error (MSE)                             |

Zastosowano predykcję kroczącą (*rolling forecast*). Wyjście modelu dla
chwili \$t+1\$ jest dołączane do wektora wejściowego w celu
wygenerowania predykcji dla chwili \$t+2\$.

**5. Wymagania Sprzętowe i Programowe**

**5.1. Środowisko uruchomieniowe**

Projekt został zaimplementowany w języku **Python 3.8+**. Do działania
wymagane są następujące biblioteki (zależności):

- tensorflow: Biblioteka backendowa do budowy i treningu sieci
  neuronowych.

- scikit-learn: Moduł MinMaxScaler do normalizacji danych.

- yfinance: Interfejs API do pobierania danych giełdowych.

- pandas / numpy: Manipulacja strukturami danych i obliczenia
  numeryczne.

- streamlit: Framework do budowy interfejsu graficznego (GUI).

- matplotlib: Wizualizacja wyników.

**5.2. Instrukcja uruchomienia**

W celu uruchomienia aplikacji w środowisku lokalnym należy wykonać
polecenie w terminalu:

streamlit run app.py

**6. Wyniki i Ewaluacja**

System prezentuje wyniki w formie graficznej, nanosząc na wykres:

1.  Rzeczywisty przebieg kursu (dane historyczne).

2.  Punkty zidentyfikowane jako anomalie (oznaczone kolorem czerwonym).

3.  Prognozę wygenerowaną przez sieć LSTM (oznaczoną kolorem zielonym).

Ewaluacja jakości modelu Autoenkodera odbywa się poprzez analizę
histogramu błędów rekonstrukcji oraz wizualną ocenę poprawności detekcji
nagłych skoków cenowych. Jakość predykcji LSTM zależy od horyzontu
czasowego -- ze względu na iteracyjny charakter prognozy, błąd rośnie
wraz z długością prognozowanego okresu.

**7. Wnioski**

Zastosowanie architektury hybrydowej pozwoliło na zwiększenie odporności
modelu predykcyjnego na szum rynkowy. Wstępne oczyszczenie danych przez
Autoenkoder zapobiega \"przeuczeniu\" się sieci LSTM na zdarzeniach
nietypowych, co w teorii prowadzi do lepszej generalizacji trendu
głównego.
