---
name: czech-company-research
description: Syntetizace rešerše českých firem z registrů, veřejných webů, výročních zpráv, médií a dalších veřejných důkazů s citacemi.
license: MIT
compatibility: Vyžaduje Python 3.12, Microsoft Agent Framework a přístup k veřejnému webu.
---

# Rešerše české firmy

Tuto dovednost použij při syntéze rešerše české firmy ze strukturovaných důkazů. Výstup piš česky.

Pravidla:

1. Data z registrů vázaná na IČO považuj za nejspolehlivější zdroj identity.
2. Neslučuj důkazy o podobně pojmenovaných firmách, pokud shodu nepodporuje IČO, doména, adresa nebo přesná obchodní firma.
3. Každé podstatné tvrzení musí citovat jedno nebo více citačních ID ze vstupu.
4. Odděluj fakta od interpretace. Sentiment a rizika označuj jako hodnocení.
5. Pokud si důkazy odporují, konflikt výslovně popiš místo tichého výběru jedné varianty.
6. Pokud data chybí, uveď, co nebylo nalezeno a jaké typy zdrojů byly zkontrolovány.

Preferovaný výstup:

- Exekutivní shrnutí
- Identita firmy
- Produkty a podnikatelská činnost
- Finanční a dokumentové signály
- Zprávy a oznámení
- Rizikové a právní signály
- Otevřené otázky a doporučené navazující zdroje
