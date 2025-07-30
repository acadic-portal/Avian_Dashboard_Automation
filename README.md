# Updating the Avian Influenza Forecasting Dashboard

In this repository a workflow is set up to update our [avian influenza forecasting dashboard](https://aimmlab.org/early-warning-system-for-avian-influenza-outbreaks) every 4 days. The python file - UpdateData.py - is composed of 4 parts:
- Collecting data from 4 different sources:
	> [Avian influenza cases in wild birds in Canada](https://www.arcgis.com/apps/dashboards/89c779e98cdf492c899df23e1c38fdbc)
 > 
	> [Google Trends](https://trends.google.com/trends/explore?date=now%201-d&geo=CA&q=%2Fm%2F0292d3&hl=en)
> 
	> Google News
> 
	> Reddit
> 
	> GDELT
- Building a GRU model using PyTorch
- Training the GRU model for forecasting avian influenza in the next 14 days
- Forecasting avian influenza cases in the next 14 days
- Updating the datasets of the [avian influenza forecasting dashboard](https://aimmlab.org/early-warning-system-for-avian-influenza-outbreaks) 

The requirements file - requirements.txt -  includes all the required packages and libraries for running the python file - UpdateData.py.
The .github/workflows/ path incudes the yaml file - AvianAutomation.yml -  of the workflow.
For more information kindly refer to our manuscript:

Nia ZM, Bragazzi NL, Gizo I, Gillies M, Gardner E, Leung D, Kong JD, Integrating Deep Learning Methods and Web-Based Data Sources for Surveillance, Forecasting, and Early Warning of Avian Influenza, SSRN, 2025; doi: https://dx.doi.org/10.2139/ssrn.5253151
