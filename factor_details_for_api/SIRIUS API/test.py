from sirius import Sirius
from config_sirius import CONFIG
import pandas as pd
from statsmodels.formula.api import ols
import os

sirius = Sirius(CONFIG)

lab = sirius.factor_lab()

Market_cap = lab.get_market_cap_for_period('20101231', '20191231')
equity = lab.get_item_for_period('ifrs_Equity', '20101231', '20191231')
BEME = equity/Market_cap[0]
Market_cap = lab.get_market_cap_for_period('20110630', '20200630')

# SMB
SMB_mask = lab.get_mask(Market_cap[0], lab.get_market_category(Market_cap[0]), market_bp = 'KOSPI')
SMB = lab.get_factor_on_date_by_mask(SMB_mask, term = 'm', winsorize_limits = 0.01, weight = 'VW')
# HML
BEME_shifted = lab.shift_date_quarter(BEME, 2)
HML_mask = lab.get_mask(BEME_shifted, lab.get_market_category(BEME_shifted), market_bp = 'KOSPI', breakpoint = [0.3, 0.7])
HML = lab.get_factor_on_date_by_mask(HML_mask, term = 'm', winsorize_limits = 0.01, weight = 'VW')

# RMW 계산하기
# (영업이익 - 이자비용) / 총자산
Total_asset = lab.get_item_for_period('ifrs_Assets', '20101231', '20191231')
fcost1 = lab.get_item_for_period('ifrs_FinanceCosts', '20101231', '20191231')
fcost2 = lab.get_item_for_period('dart_InterestExpenseFinanceExpense', '20101231', '20191231')
fcost = pd.concat([fcost1, fcost2], axis = 1).groupby(level=0, axis=1).last()
OI = lab.get_item_for_period('dart_OperatingIncomeLoss', '20101231', '20191231')

RMW_base = (OI - fcost) / Total_asset
RMW_base_shifted = lab.shift_date_quarter(RMW_base, 2)
RMW_mask = lab.get_mask(RMW_base_shifted, lab.get_market_category(RMW_base_shifted), market_bp = 'KOSPI', breakpoint = [0.3, 0.7])
RMW = lab.get_factor_on_date_by_mask(RMW_mask, term = 'm', winsorize_limits = 0.01, weight = 'VW')

# CMA 계산하기

Total_asset = lab.get_item_for_period('ifrs_Assets', '20091231', '20181231')
pnl = lab.get_item_for_period('ifrs_ProfitLoss', '20101231', '20191231')
Total_asset_shifted = lab.shift_date_quarter(Total_asset, 4)
CMA_base = pnl / Total_asset_shifted
CMA_base_shifted = lab.shift_date_quarter(CMA_base, 2)
CMA_mask = lab.get_mask(CMA_base_shifted, lab.get_market_category(CMA_base_shifted), market_bp = 'KOSPI', breakpoint = [0.3, 0.7])
CMA = lab.get_factor_on_date_by_mask(CMA_mask, term = 'm', winsorize_limits = 0.01, weight = 'VW')

cd_91 = pd.read_csv('cd금리.csv', encoding='cp949', header=2, index_col=0).T.iloc[:, [4]]
cd_91.index = cd_91.index.map(lambda x: x[:6])
cd_91.columns = ['cd_91']

factors = pd.concat([(lab.get_risk_free_rate(SMB, cd_91).astype(float)/12).squeeze(), SMB.iloc[:,0] - SMB.iloc[:,1], HML.iloc[:,2]- HML.iloc[:,0], RMW.iloc[:,2] - RMW.iloc[:,0], CMA.iloc[:,0] - CMA.iloc[:,2]], axis=1, join='inner')
factors.columns = ['Rm', 'SMB', 'HML', 'RMW', 'CMA']

HMLO_ols = ols(formula = 'HML ~ Rm+SMB+RMW+CMA', data = factors).fit()
beta_Rm = HMLO_ols.params['Rm']
beta_SMB = HMLO_ols.params['SMB']
beta_RMW = HMLO_ols.params['RMW']
beta_CMA = HMLO_ols.params['CMA']
HMLO = pd.DataFrame(factors["HML"] - beta_Rm * factors['Rm'].squeeze() - beta_SMB*factors['SMB'].squeeze() - beta_RMW*factors['RMW'].squeeze() - beta_CMA*factors['CMA'].squeeze())
HMLO.columns = ['HMLO']
factors = pd.concat([factors, HMLO], axis=1)

# HML pfo
hml_pfo_mask = lab.make_portfolio(lab.get_mask(Market_cap[0], Market_cap[1], breakpoint = [0.2, 0.4, 0.6, 0.8], market_bp='KOSPI'), BEME_shifted, breakpoint = [0.2, 0.4, 0.6, 0.8])
pfo25 = lab.serialize_pfo([lab.get_mask(Market_cap[0], Market_cap[1], breakpoint = [0.2, 0.4, 0.6, 0.8], market_bp='KOSPI'), hml_pfo_mask])
pfo_return = lab.get_factor_on_date_by_mask(pfo25, term = 'm', winsorize_limits= 0.01, weight = 'VW')

coef = []
tvalues = []
pvalues = []
se = []
r = []
for i in range(25):
    model = lab.get_factor_on_date_by_maskget_ols_model(pfo25, pfo_return.iloc[:,i], cd_91, SMB.iloc[:,0] - SMB.iloc[:,1], HML.iloc[:,2] - HML.iloc[:,0], RMW.iloc[:,2] - RMW.iloc[:,0], CMA.iloc[:,0] - CMA.iloc[:,2])
    coef.append(model.params)
    tvalues.append(model.tvalues)
    pvalues.append(model.pvalues)
    se.append(model.bse)
    r.append(model.rsquared_adj)

print(pd.concat(coef, axis=1))
