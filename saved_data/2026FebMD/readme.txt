This folder contains data taken on a single microdiamond mounted on a FlexPCB stripline. Temperature is controlled by a Thorlabs Flexible Resistive Foil Heaters (HT10K) attached to the bottom of the PCB. Note that an offset is expected between the set temperature and the actual temperature at the sample. 

TEC_cal.xlsx contains calibration data for the heater. The read temperature is measured on the PCB adjacent to the microdiamond and may not be the true temperature at the sample. This measurement is to show the linearity between set and read temperature.

ODMR spectra are taken using a confocal microscope. Temperature is set to be between 25 and 45 degC. Measurement is done every 5 degC.
Each ODMR file contains 25 single measurements. Averaging over 5 measurements gives a good SNR. 

Cycle 1: ODMR spectra were taken at temperature 25, 30, 35, 40, 45 degC. At 45 degC, two magnetic field measurements were taken with the permanent magnets 130 and 90 mm apart.
	The temperature was then ramped down. More ODMR spectra are obtained at 45, 40, 35, 30, 25 degC. At 25 degC, the magnets are installed again and measurements were taken at 	25 and 35 degC. The each magnetic field file only contains 5 single measurements

Cycle 2: ODMR spectra were taken at temperature 25, 30, 35, 40, 45, 40, 35, 30, 25 degC.

Data files are formatted as the following:
frequency (MHz)	  PL_sig1  PL_ref1   PL_sig2  PL_ref2  ...  avgPL_sig   avgPL_ref


To calculate Pl change for ODMR: \DeltaPL = PL_sig/PL_ref  