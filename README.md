</head>
<body>
  <h1>Univariate & Multivariate Analysis Toolkit</h1>

  <p>
    This project provides a Python script to perform <strong>comprehensive univariate and multivariate analyses</strong> 
    on numeric datasets. It automatically generates <strong>histograms</strong>, <strong>KDEs</strong>, <strong>boxplots</strong>, 
    <strong>descriptive statistics</strong>, <strong>pairplots for top predictors</strong>, and <strong>OLS regression</strong> 
    with coefficients, t-values, and p-values.
  </p>

  <h2>Statistical Background</h2>
  <p>
    The script implements statistical measures useful for <strong>machine learning preprocessing</strong>. 
    All equations and explanations use proper subscripts, superscripts, and Greek letters for clarity.
  </p>

  <p><strong>Mean (average) of a variable X:</strong></p>
  <p>$$
  \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$X_i$$ = ith observation of variable X</li>
    <li>$$\bar{X}$$ = mean value of X</li>
    <li>$$n$$ = total number of observations</li>
  </ul>

  <p><strong>Standard deviation:</strong></p>
  <p>$$
  \sigma_X = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2}
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$\sigma_X$$ = standard deviation of X</li>
    <li>$$X_i$$ = ith observation</li>
    <li>$$\bar{X}$$ = mean of X</li>
    <li>$$n$$ = total number of observations</li>
  </ul>

  <p><strong>Z-score standardization:</strong></p>
  <p>$$
  Z_i = \frac{X_i - \bar{X}}{\sigma_X}
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$Z_i$$ = standardized value of X_i</li>
    <li>$$X_i$$ = original observation</li>
    <li>$$\bar{X}$$ = mean of X</li>
    <li>$$\sigma_X$$ = standard deviation of X</li>
  </ul>

  <p><strong>Covariance between X and Y:</strong></p>
  <p>$$
  \text{Cov}(X,Y) = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$\text{Cov}(X,Y)$$ = covariance between X and Y</li>
    <li>$$X_i, Y_i$$ = ith observations of X and Y</li>
    <li>$$\bar{X}, \bar{Y}$$ = means of X and Y</li>
    <li>$$n$$ = total number of observations</li>
  </ul>

  <p><strong>Pearson Correlation Coefficient (r):</strong></p>
  <p>$$
  r_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$r_{XY}$$ = Pearson correlation coefficient between X and Y</li>
    <li>Value ranges from -1 to 1</li>
  </ul>

  <p><strong>OLS Regression (Multivariate):</strong></p>
  <p>$$
  Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$Y$$ = dependent variable (target)</li>
    <li>$$X_1, \dots, X_p$$ = predictor variables</li>
    <li>$$\beta_0$$ = intercept, $$\beta_i$$ = regression coefficients</li>
    <li>$$\epsilon$$ = error term</li>
  </ul>

  <p><strong>t-statistic for regression coefficient:</strong></p>
  <p>$$
  t_i = \frac{\hat{\beta}_i}{SE(\hat{\beta}_i)}, \quad df = n-p-1
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$\hat{\beta}_i$$ = estimated coefficient</li>
    <li>$$SE(\hat{\beta}_i)$$ = standard error of coefficient</li>
    <li>$$df$$ = degrees of freedom = n - p - 1</li>
  </ul>

  <p><strong>R-squared (Coefficient of Determination):</strong></p>
  <p>$$
  R^2 = 1 - \frac{\sum_{i=1}^n (Y_i - \hat{Y}_i)^2}{\sum_{i=1}^n (Y_i - \bar{Y})^2}
  $$</p>
  <p>Explanation:</p>
  <ul>
    <li>$$R^2$$ = proportion of variance in Y explained by the model</li>
    <li>$$Y_i$$ = observed values, $$\hat{Y}_i$$ = predicted values, $$\bar{Y}$$ = mean of Y</li>
  </ul>

  <h2>Features</h2>
  <ul>
    <li><strong>Automatic Preprocessing:</strong> Detects numeric predictors and target, handling missing data.</li>
    <li><strong>Univariate Analysis:</strong> Histograms, KDEs, boxplots, and descriptive statistics.</li>
    <li><strong>Multivariate Analysis:</strong> Pairplots of top predictors and OLS regression with full summary.</li>
    <li><strong>High-Resolution Outputs:</strong> PNG, CSV, and text summaries ready for reports and ML workflows.</li>
  </ul>

  <h2>Usage Instructions</h2>
  <ol>
    <li>Place your numeric dataset in the project directory.</li>
    <li>Run the script with Python:
      <pre><code>python Univariate_Multivariate_Analysis.py</code></pre>
    </li>
    <li>The script will:
      <ul>
        <li>Perform univariate analysis for all numeric columns</li>
        <li>Generate pairplots for top predictors against the target</li>
        <li>Run multivariate OLS regression and save coefficients, t-values, and R²</li>
        <li>Export all results to CSV, PNG, and text files</li>
      </ul>
    </li>
  </ol>

  <h2>Requirements</h2>
  <ul>
    <li>Python 3.8+</li>
    <li>Libraries: pandas, numpy, matplotlib, seaborn, statsmodels, scipy</li>
    <li>Numeric dataset with at least two non-empty columns</li>
  </ul>

  <h2>License</h2>
  <p>This project is licensed under the <strong>MIT License</strong>. Feel free to use, modify, and redistribute. Credit is appreciated.</p>

  <h2>Developer Info</h2>
  <ul>
    <li><strong>Developer:</strong> Engr. Tufail Mabood, MSC STRUCTURE UET PESHAWAR</li>
    <li><strong>Contact:</strong> <a href="https://wa.me/+923440907874">WhatsApp</a></li>
    <li><strong>Note:</strong> If you need help with statistical analysis or data preprocessing, feel free to reach out.</li>
  </ul>
</body>
