# pm-AMM Simulator

Um simulador interativo para mercados de previsão usando o **pm-AMM** (Prediction Market AMM), um AMM uniforme otimizado para mercados de previsão desenvolvido pela [Paradigm](https://www.paradigm.xyz/2024/11/pm-amm).

## 🚀 Funcionalidades

### Tipos de AMM Suportados

O simulador suporta duas variantes do pm-AMM:

#### Static pm-AMM
- Liquidez constante ao longo do tempo
- Invariante: `(y - x) Φ((y-x)/L) + L φ((y-x)/L) - y = 0`
- Ideal para mercados onde a volatilidade não aumenta significativamente perto da expiração
- **Trades não possuem timestamp** (dia/hora são ignorados)

#### Dynamic pm-AMM
- Liquidez que diminui conforme o mercado se aproxima da expiração
- Invariante: `(y - x) Φ((y-x)/(L√(T-t))) + L√(T-t) φ((y-x)/(L√(T-t))) - y = 0`
- Mantém LVR (Loss-vs-Rebalancing) constante ao longo do tempo
- Protege LPs de perdas aumentadas perto da expiração
- **Cada trade possui dia e hora**, afetando a liquidez efetiva

### Parâmetros Configuráveis
- **L (Liquidity Parameter)**: Parâmetro de escala/liquidez do pm-AMM
- **Base Fee Rate**: Taxa aplicada em cada trade
- **Market Duration (days)** (Dynamic pm-AMM): Duração total do mercado em dias

### Gerenciamento de Trades

Cada trade no modo **Dynamic** possui:
- `direction`: YES ou NO
- `shares`: quantidade de shares
- `day`: dia da trade (1 a duração do mercado)
- `time`: hora da trade no formato "HH:MM"

No modo **Static**, os campos `day` e `time` são ignorados.

### Funcionalidades de Trades
- ✅ Adicionar trades individualmente com dia/hora
- ✅ Importar trades em lote via JSON
- ✅ Visualizar todas as trades em tabela interativa
- ✅ Ordenação automática por dia/hora
- ✅ Cálculo dinâmico de T-t para cada trade

### Simulação e Resultados
- Cálculo automático de custos usando fórmulas pm-AMM
- **L_eff dinâmico**: liquidez efetiva calculada para cada trade
- Gráfico de evolução de preços ao longo do tempo
- Gráfico de evolução do L_eff (Dynamic pm-AMM)
- Visualização da curva invariante do AMM
- Resumo financeiro completo

## 📦 Instalação

### Requisitos
- Python 3.7+
- Streamlit
- NumPy
- Pandas
- Matplotlib
- SciPy

### Instalação das Dependências

```bash
pip install -r requirements.txt
```

## 🎯 Como Usar

### Executar a Aplicação

```bash
streamlit run streamlit_app.py
```

### Selecionar Tipo de AMM

1. Na seção "AMM Type", escolha entre **Static** ou **Dynamic**
2. Para Dynamic, configure a **duração do mercado em dias**

### Adicionar Trades Manualmente

**Modo Dynamic:**
1. Selecione o **Dia** (1 até duração do mercado)
2. Selecione a **Hora** (formato HH:MM)
3. Escolha a direção (YES ou NO)
4. Defina a quantidade de Shares
5. Clique em "Add Trade"

**Modo Static:**
1. Escolha a direção (YES ou NO)
2. Defina a quantidade de Shares
3. Clique em "Add Trade"

### Importar Trades via JSON

1. Clique em "JSON Template" para ver o formato esperado
2. Clique em "Import JSON"
3. Cole o JSON no campo de texto
4. Clique em "Confirm Import"

#### Formato JSON

```json
{
  "trades": [
    {"direction": "YES", "shares": 10, "day": 1, "time": "09:30"},
    {"direction": "NO", "shares": 5, "day": 3, "time": "14:15"},
    {"direction": "YES", "shares": 20, "day": 7, "time": "17:49"},
    {"direction": "NO", "shares": 15, "day": 12, "time": "10:00"}
  ]
}
```

**Campos:**
- `direction`: "YES" ou "NO" (obrigatório)
- `shares`: quantidade de shares (obrigatório)
- `day`: dia da trade, 1 a N (opcional, default: 1)
- `time`: hora no formato "HH:MM" (opcional, default: "12:00")

⚠️ No modo **Static**, os campos `day` e `time` são ignorados durante a simulação.

## 🔁 Polymarket Portability (Off-chain CLOB Simulation)

This branch adds an **off-chain pm-AMM portability layer** for Polymarket-style CLOB execution:

- `polymarket_port/pmamm_math.py` → reusable pm-AMM math utilities
- `polymarket_port/clob_simulator.py` → buy-only YES/NO routing simulation against bid/ask
- `scripts/run_polymarket_port_iterations.py` → iterative parameter runs + outcome simulations

### Run iterative simulations (synthetic tape)

```bash
python scripts/run_polymarket_port_iterations.py \
  --iterations 8 \
  --trials-per-iteration 200 \
  --out-dir outputs/polymarket_port
```

### Run iterative simulations on **real Polymarket data** (paper buys)

```bash
python scripts/run_polymarket_realdata_iterations.py \
  --slug btc-updown-15m-1771409700 \
  --iterations 10 \
  --trials-per-iteration 150 \
  --window-steps 60 \
  --out-dir outputs/polymarket_port_real
```

Or auto-discover a hot market from latest trades:

```bash
python scripts/run_polymarket_realdata_iterations.py \
  --query btc-updown-15m \
  --iterations 10 \
  --trials-per-iteration 150
```

Outputs:

- `real_tape_*.csv` (bucketed real-market tape used for simulation)
- `raw_trades_*.csv` (filtered trade prints for auditability)
- `iteration_summary_*.csv` (one row per parameter iteration)
- `iteration_trials_*.csv` (per-trial details)
- `best_iteration_*.json` (top candidate configuration)

### Notes

- This is designed for **porting to Polymarket execution bots** (off-chain), not direct smart-contract deployment.
- Trading logic is intentionally **buy-only YES/NO-compatible** with CLOB routing semantics.
- The pm-AMM state acts as a **virtual fair-price model** while paper fills are simulated from market data.
- For historical runs, bid/ask per bucket are inferred from mid + spread proxy (configurable / live-estimated).

## 📊 Entendendo os Resultados

### Cálculo do Tempo até Expiração (T-t)

Para o Dynamic pm-AMM, o tempo até expiração é calculado como:

```
T-t = market_duration_days - elapsed_days
elapsed_days = (day - 1) + (hour + minute/60) / 24
```

**Exemplo:** Mercado de 14 dias
- Trade no Dia 1 às 00:00 → T-t = 14.0 dias
- Trade no Dia 1 às 12:00 → T-t = 13.5 dias
- Trade no Dia 7 às 18:00 → T-t = 7.25 dias
- Trade no Dia 14 às 23:59 → T-t ≈ 0.001 dias

### Métricas Exibidas

**Tabela de Trades (Dynamic):**
- **Day/Time**: Quando a trade ocorreu
- **T-t (days)**: Tempo restante até expiração
- **L_eff**: Liquidez efetiva = L × √(T-t)
- **Price Before/After**: Preços antes e depois da trade
- **Avg. Price**: Preço médio por share
- **Cost Paid**: Custo da trade
- **Fee**: Taxa cobrada

### Visualizações

- **Price Evolution**: Gráfico mostrando como os preços YES/NO evoluem ao longo do tempo
- **L_eff Evolution** (Dynamic): Gráfico de barras mostrando como a liquidez efetiva diminui
- **Invariant Curve**: Visualização da curva invariante do pm-AMM

## 🔧 Tecnologias Utilizadas

- **Streamlit**: Framework para aplicações web interativas
- **NumPy**: Cálculos numéricos
- **Pandas**: Manipulação de dados
- **Matplotlib**: Visualizações e gráficos
- **SciPy**: Funções estatísticas (distribuição normal) e otimização numérica

## 📝 Sobre o pm-AMM

O **pm-AMM** é um AMM (Automated Market Maker) uniforme desenvolvido especificamente para mercados de previsão.

### Static vs Dynamic pm-AMM

| Aspecto | Static | Dynamic |
|---------|--------|---------|
| Liquidez (L_eff) | Constante = L | Diminui = L × √(T-t) |
| LVR | Aumenta perto da expiração | Constante ao longo do tempo |
| Uso de tempo | Não utiliza | Cada trade tem dia/hora |
| Ideal para | Mercados de longo prazo | Mercados com data de expiração definida |

### Fórmulas Matemáticas

**Preço YES:**
```
P = Φ((y - x) / L_eff)
```

**Valor do Portfolio:**
```
V(P) = L_eff × φ(Φ⁻¹(P))
```

**Liquidez Efetiva:**
```
L_eff = L           (static)
L_eff = L × √(T-t)  (dynamic)
```

Onde:
- `x`, `y` são as reservas de tokens YES e NO
- `L` é o parâmetro de liquidez base
- `T-t` é o tempo restante até expiração
- `φ` é a PDF da distribuição normal padrão
- `Φ` é a CDF da distribuição normal padrão

## 📚 Referências

- [pm-AMM: A Uniform AMM for Prediction Markets](https://www.paradigm.xyz/2024/11/pm-amm) - Paradigm Research

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👤 Autor

**Iago Macedo**

---

Desenvolvido com ❤️ para simulação de mercados de previsão
