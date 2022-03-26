********************************************************************
* NetworkTMLE: Comparing Model Outputs from SAS
********************************************************************;

DATA qmodel; *Data input;
	INPUT w a y a_map w_map degree c_;
	DATALINES;
1 1 1 2 1 3 1
0 0 0 2 2 3 -1
0 1 0 1 1 3 5
0 0 1 2 0 2 0
1 0 0 2 1 2 0
1 0 1 0 0 1 0
0 1 0 0 1 2 10
0 0 0 0 0 0 -5
1 1 0 1 2 2 -5
;
RUN;
DATA qmodel;
	SET qmodel;
	c = (c_ - -5.0005) / (10.0005 + 5.0005);
RUN;

* Logit Q-model;
PROC GENMOD DATA=qmodel DESC;
	MODEL y = a a_map w / LINK=logit DIST=b; 
	OUTPUT out=rnetpred p=prediction;
RUN;

* Linear Q-model;
PROC GENMOD DATA=qmodel DESC;
	MODEL c = a a_map w / LINK=identity DIST=n; 
	OUTPUT out=rnetpred p=prediction;
RUN;

* Poisson Q-model;
PROC GENMOD DATA=qmodel DESC;
	MODEL c = a a_map w / LINK=log DIST=poisson; 
	OUTPUT out=rnetpred p=prediction;
RUN;

* Logit Q-model -- restricted;
PROC GENMOD DATA=qmodel DESC;
	MODEL y = a_map / LINK=logit DIST=b; 
	OUTPUT out=rnetpred p=prediction;
	WHERE degree < 3;
RUN;

* Linear Q-model -- restricted;
PROC GENMOD DATA=qmodel DESC;
	MODEL c = a a_map w / LINK=identity DIST=n; 
	OUTPUT out=rnetpred p=prediction;
	WHERE degree < 3;
RUN;

* Poisson Q-model -- restricted;
PROC GENMOD DATA=qmodel DESC;
	MODEL c = a a_map / LINK=log DIST=poisson; 
	OUTPUT out=rnetpred p=prediction;
	WHERE degree < 3;
RUN;

* gi-model;
PROC GENMOD DATA=qmodel DESC;
	MODEL a = w w_map / LINK=log DIST=poisson; 
	OUTPUT out=rnetpred p=prediction;
RUN;
PROC PRINT DATA=rnetpred;
RUN;

*******************************************
* Analyzing R processed data
*******************************************;
* imported tmlenet_r_processed.csv file;

* Q-model;
PROC GENMOD DATA=dat DESC;
	MODEL y = a w a_sum w_sum / LINK=logit DIST=b; 
RUN;

* g-models;
PROC GENMOD DATA=dat DESC;
	MODEL a = w w_sum / LINK=logit DIST=b; 
RUN;
PROC GENMOD DATA=dat DESC;
	MODEL a_map1 = a w w_sum / LINK=logit DIST=b; 
RUN;
PROC GENMOD DATA=dat DESC;
	MODEL a_map2 = a w w_sum a_map1 / LINK=logit DIST=b; 
RUN;

* Poisson gs-model;
PROC GENMOD DATA=dat DESC;
	MODEL a_sum = a w w_sum / LINK=log DIST=poisson; 
RUN;
PROC GENMOD DATA=dat DESC;
	MODEL a_mean = a w w_sum / LINK=log DIST=poisson; 
RUN;

* Multinomial gs-model;
PROC GENMOD DATA=dat DESC;
	MODEL a_sum = a w w_sum / LINK=log DIST=poisson; 
RUN;

* Linear gs-model;
PROC REG DATA=dat;
  MODEL a_sum = a w w_sum;
RUN;
PROC REG DATA=dat;
  MODEL a_mean = a w w_sum;
RUN;
