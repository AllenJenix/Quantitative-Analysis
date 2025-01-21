import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import mplfinance as mpf
from tqdm import tqdm


# MACD 지표 계산 함수
def MACD(df, window_fast, window_slow, window_signal):
    macd = pd.DataFrame(index=df.index)

    macd['ema_fast'] = df['Close'].ewm(span=window_fast, adjust=False).mean()
    macd['ema_slow'] = df['Close'].ewm(span=window_slow, adjust=False).mean()

    macd['macd'] = macd['ema_fast'] - macd['ema_slow']
    macd['signal'] = macd['macd'].ewm(span=window_signal, adjust=False).mean()
    macd['diff'] = macd['macd'] - macd['signal']  # Histogram

    macd['bar_positive'] = macd['diff'].apply(lambda x: x if x > 0 else 0)
    macd['bar_negative'] = macd['diff'].apply(lambda x: x if x < 0 else 0)
    return macd


# Buy-and-Hold 전략 누적 수익률 계산
def strategyBuyAndHold_return(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df['Cumulative_Return'] = (1 + df['Close'].pct_change()).cumprod()
    return df['Cumulative_Return']


# MACD 전략 누적 수익률 계산
def strategyMACD_return(df: pd.DataFrame, f=12, s=26, sig=9) -> pd.Series:
    df = df.copy()
    macd = MACD(df, f, s, sig)
    df['Signal'] = 0
    df.loc[macd['macd'] > macd['signal'], 'Signal'] = 1
    df['Position'] = df['Signal'].shift(1).fillna(0)

    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']

    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()

    return df['Cumulative_Strategy']


# 시계열 교차 검증 점수 함수
def time_series_forward_walk_cv_score(df: pd.DataFrame, f: int, s: int, sig: int, n_splits=5,
                                      train_window=None) -> float:
    length = len(df)
    fold_size = length // n_splits
    scores = []

    # 폴드 크기 검증
    if fold_size < 1:
        raise ValueError("데이터 길이가 너무 짧아서 지정한 n_splits로 폴드를 생성할 수 없습니다.")

    # 고정된 훈련 윈도우 크기 설정
    if train_window is None:
        train_window = fold_size  # 기본적으로 한 폴드 크기만큼 설정

    # 인덱스 중복 제거
    if not df.index.is_unique:
        print("데이터 프레임의 인덱스가 중복되어 있습니다. 중복을 제거합니다.")
        df = df[~df.index.duplicated(keep='last')]

    # 데이터 정렬
    df = df.sort_index()

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else length

        # 고정된 크기의 훈련 윈도우 설정
        train_start = max(0, test_start - train_window)
        train_end = test_start
        train_data = df.iloc[train_start:train_end].copy()
        test_data = df.iloc[test_start:test_end].copy()

        # 테스트 데이터가 비어있으면 건너뜀
        if test_data.empty:
            print(f"폴드 {i + 1}: test_data가 비어있어 건너뜁니다.")
            continue

        # 훈련 데이터에 MACD 적용
        macd_train = MACD(train_data, f, s, sig)
        train_data['Signal'] = 0
        train_data.loc[macd_train['macd'] > macd_train['signal'], 'Signal'] = 1
        train_data['Position'] = train_data['Signal'].shift(1).fillna(0)
        train_data['Daily_Return'] = train_data['Close'].pct_change()
        train_data['Strategy_Return'] = train_data['Daily_Return'] * train_data['Position']
        train_data['Cumulative_Strategy'] = (1 + train_data['Strategy_Return']).cumprod()

        # 테스트 데이터에 대해 시그널 생성
        combined_test = pd.concat([train_data, test_data])
        macd_test = MACD(combined_test, f, s, sig)

        # 테스트 데이터 시그널 설정
        try:
            test_signals = macd_test.loc[test_data.index, 'macd'] > macd_test.loc[test_data.index, 'signal']
        except KeyError as e:
            print(f"폴드 {i + 1}: test_data의 인덱스 중 일부가 macd_test에 존재하지 않습니다. {e}")
            continue

        if len(test_signals) != len(test_data):
            print(f"폴드 {i + 1}: test_signals 크기({len(test_signals)})와 test_data 크기({len(test_data)})가 일치하지 않습니다.")
            continue

        test_data['Signal'] = 0
        test_data.loc[test_signals, 'Signal'] = 1
        test_data['Position'] = test_data['Signal'].shift(1).fillna(
            train_data['Position'].iloc[-1] if not train_data['Position'].empty else 0)
        test_data['Daily_Return'] = test_data['Close'].pct_change()
        test_data['Strategy_Return'] = test_data['Daily_Return'] * test_data['Position']
        test_data['Cumulative_Strategy'] = (1 + test_data['Strategy_Return']).cumprod()

        # 누적 수익 계산
        if train_data['Cumulative_Strategy'].empty:
            start_val = 1.0
        else:
            start_val = train_data['Cumulative_Strategy'].iloc[-1]

        if 'Cumulative_Strategy' in test_data.columns and not test_data['Cumulative_Strategy'].empty:
            end_val = test_data['Cumulative_Strategy'].iloc[-1]
        else:
            print(f"폴드 {i + 1}: 'Cumulative_Strategy'가 비어있어 건너뜁니다.")
            continue

        fold_return = end_val / start_val - 1
        scores.append(fold_return)

    if not scores:
        return 0.0  # 모든 폴드가 비어있을 경우 기본값 반환

    return np.mean(scores)


# 최적화 함수
def optimizeMACD(df: pd.DataFrame):
    param_fast = range(10, 180, 10)
    param_slow = range(25, 395, 10)
    param_signal = range(10, 130, 10)

    best_params = None
    best_return = -np.inf

    loop_count = 0

    for f in tqdm(param_fast, desc='Searching MACD', total=len(param_fast)):
        for s in param_slow:
            if f >= s:
                continue
            for sig in param_signal:
                cv_score = time_series_forward_walk_cv_score(df, f, s, sig, n_splits=5)

                if cv_score > best_return:
                    best_return = cv_score
                    best_params = (f, s, sig)

                loop_count += 1

    return best_params, best_return


# 최적화된 MACD 전략 누적 수익률 계산
def strategyOptMACD_return(df: pd.DataFrame, f: int, s: int, sig: int) -> pd.Series:
    df = df.copy()
    macd_df = MACD(df, f, s, sig)

    df['Signal'] = 0
    df.loc[macd_df['macd'] > macd_df['signal'], 'Signal'] = 1
    df['Position'] = df['Signal'].shift(1).fillna(0)

    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']

    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()

    return df['Cumulative_Strategy']


# 전략별 누적 수익률 계산 함수
def calculate_strategies(df_weekly: pd.DataFrame):
    buy_hold = strategyBuyAndHold_return(df_weekly)
    macd = strategyMACD_return(df_weekly, f=12, s=26, sig=9)
    return buy_hold, macd


# 플롯 및 저장 함수 정의
def plot_and_save_backtest(
        buy_hold: pd.Series,
        macd: pd.Series,
        opt_macd: pd.Series,
        macd_basic: pd.DataFrame,
        macd_opt: pd.DataFrame,
        title: str,
        save_filename: str
):
    """
    전략별 누적 수익률과 MACD 지표를 시각화하고 PNG 파일로 저장하는 함수.

    Parameters:
    - buy_hold (pd.Series): Buy-and-Hold 전략의 누적 수익률.
    - macd (pd.Series): MACD 전략의 누적 수익률.
    - opt_macd (pd.Series): 최적화된 MACD 전략의 누적 수익률.
    - macd_basic (pd.DataFrame): 기본 MACD 지표 데이터프레임.
    - macd_opt (pd.DataFrame): 최적화된 MACD 지표 데이터프레임.
    - title (str): 플롯의 제목.
    - save_filename (str): 저장할 PNG 파일의 이름 (현재 디렉토리에 저장).
    """
    # 플롯 크기 및 레이아웃 설정
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.4)

    # 상단 서브플롯: 전략별 누적 수익률
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(buy_hold.index, buy_hold, label='Buy and Hold', color='black')
    ax1.plot(macd.index, macd, label='MACD Strategy', color='blue')
    ax1.plot(opt_macd.index, opt_macd, label='Optimized MACD Strategy', color='green')
    ax1.set_title(f'{title} Strategy Cumulative Returns')
    ax1.set_ylabel('Cumulative Return (%)')  # Y축 레이블 수정
    ax1.legend()
    ax1.grid(True)

    # Y축을 퍼센트 형식으로 설정
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    # 중간 서브플롯: 기본 MACD 지표
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(macd_basic.index, macd_basic['macd'], label='MACD', color='blue')
    ax2.plot(macd_basic.index, macd_basic['signal'], label='Signal', color='purple')
    ax2.bar(macd_basic.index, macd_basic['bar_positive'], color='#4dc790', label='MACD Histogram Positive')
    ax2.bar(macd_basic.index, macd_basic['bar_negative'], color='#fd6b6c', label='MACD Histogram Negative')
    ax2.set_title('Basic MACD')
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True)

    # 하단 서브플롯: 최적화된 MACD 지표
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(macd_opt.index, macd_opt['macd'], label='Optimized MACD', color='green')
    ax3.plot(macd_opt.index, macd_opt['signal'], label='Optimized Signal', color='orange')
    ax3.bar(macd_opt.index, macd_opt['bar_positive'], color='#4dc790', label='Optimized MACD Histogram Positive')
    ax3.bar(macd_opt.index, macd_opt['bar_negative'], color='#fd6b6c', label='Optimized MACD Histogram Negative')
    ax3.set_title('Optimized MACD')
    ax3.set_ylabel('MACD')
    ax3.legend()
    ax3.grid(True)

    # 플롯 저장
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')

    # 플롯 닫기 (메모리 해제를 위해)
    plt.close()
    print("Plot 생성 완료")


# Download data
# SP = yf.download('^GSPC', period='5y', interval='1d')
# USLM = yf.download('USLM', period='5y', interval='1d')

# yahoo finance의 최신 이슈로 생기는 불필요한 데이터 필드 삭제
# SP.columns = SP.columns.droplevel("Ticker")
# USLM.columns = USLM.columns.droplevel("Ticker")

# Data file download
# SP.to_csv("./data/SP.csv")
# USLM.to_csv("./data/USLM.csv")

# 데이터 로드
SP = pd.read_csv("./data/SP.csv", index_col='Date', parse_dates=True)
USLM = pd.read_csv("./data/USLM.csv", index_col='Date', parse_dates=True)

# 주간 단위로 리샘플링
SP_Weekly = SP.resample('W').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})
SP_Weekly.dropna(inplace=True)

USLM_Weekly = USLM.resample('W').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})
USLM_Weekly.dropna(inplace=True)

# Chart S&P500
macd = MACD(SP_Weekly, 12, 26, 9)

macd_plot = [
    mpf.make_addplot((macd['macd']), color='blue', panel=2, ylabel='MACD', secondary_y=False),
    mpf.make_addplot((macd['signal']), color='purple', panel=2, secondary_y=False),
    mpf.make_addplot((macd['bar_positive']), type='bar', color='#4dc790', panel=2),
    mpf.make_addplot((macd['bar_negative']), type='bar', color='#fd6b6c', panel=2)
]

mpf.plot(SP_Weekly, type='candle', volume=True, addplot=macd_plot,
         savefig=dict(fname='./plot/SP_MACD.png', dpi=300, bbox_inches='tight'))

# Chart USLM
macd = MACD(USLM_Weekly, 12, 26, 9)

macd_plot = [
    mpf.make_addplot((macd['macd']), color='blue', panel=2, ylabel='MACD', secondary_y=False),
    mpf.make_addplot((macd['signal']), color='purple', panel=2, secondary_y=False),
    mpf.make_addplot((macd['bar_positive']), type='bar', color='#4dc790', panel=2),
    mpf.make_addplot((macd['bar_negative']), type='bar', color='#fd6b6c', panel=2)
]

mpf.plot(USLM_Weekly, type='candle', volume=True, addplot=macd_plot,
         savefig=dict(fname='./plot/USLM_MACD.png', dpi=300, bbox_inches='tight'))

# 최적화 수행 및 파라미터 저장
best_params_SP, best_ret_SP = optimizeMACD(SP_Weekly)
print("S&P500 최적 파라미터:", best_params_SP)
best_params_USLM, best_ret_USLM = optimizeMACD(USLM_Weekly)
print("USLM 최적 파라미터:", best_params_USLM)

# 전략별 누적 수익률 계산 (Buy-and-Hold 및 기본 MACD)
buy_hold_SP, macd_SP = calculate_strategies(SP_Weekly)
buy_hold_USLM, macd_USLM = calculate_strategies(USLM_Weekly)

# 최적화된 MACD 전략 누적 수익률 계산 (최적화된 파라미터 사용)
opt_macd_SP = strategyOptMACD_return(SP_Weekly, *best_params_SP)
opt_macd_USLM = strategyOptMACD_return(USLM_Weekly, *best_params_USLM)

# MACD 지표 계산 (이미 최적화된 파라미터 사용)
macd_basic_SP = MACD(SP_Weekly, 12, 26, 9)
macd_opt_SP = MACD(SP_Weekly, best_params_SP[0], best_params_SP[1], best_params_SP[2])

macd_basic_USLM = MACD(USLM_Weekly, 12, 26, 9)
macd_opt_USLM = MACD(USLM_Weekly, best_params_USLM[0], best_params_USLM[1], best_params_USLM[2])

# 시각화 및 저장
plot_and_save_backtest(
    buy_hold=buy_hold_SP,
    macd=macd_SP,
    opt_macd=opt_macd_SP,
    macd_basic=macd_basic_SP,
    macd_opt=macd_opt_SP,
    title='S&P500',
    save_filename='./plot/S&P500_Strategies_Backtesting.png'
)

plot_and_save_backtest(
    buy_hold=buy_hold_USLM,
    macd=macd_USLM,
    opt_macd=opt_macd_USLM,
    macd_basic=macd_basic_USLM,
    macd_opt=macd_opt_USLM,
    title='USLM',
    save_filename='./plot/USLM_Strategies_Backtesting.png'
)