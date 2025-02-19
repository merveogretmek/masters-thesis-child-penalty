# 06 App

import os
import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def load_csv(filename):
    filepath = os.path.join("results", "heterogeneous_analysis", filename)
    return pd.read_csv(filepath)

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# -------------------------
# Define a common Navigation Bar
# -------------------------
def navbar():
    return html.Nav([
        html.A("Introduction", href="/", style={"margin": "0 15px", "color": "white", "textDecoration": "none"}),
        html.A("Methodology", href="/methodology", style={"margin": "0 15px", "color": "white", "textDecoration": "none"}),
        html.A("Heterogeneous Analysis", href="/heterogeneous", style={"margin": "0 15px", "color": "white", "textDecoration": "none"}),
        html.A("Reform Analysis", href="/reform", style={"margin": "0 15px", "color": "white", "textDecoration": "none"}),
        html.A("Build Your Profile", href="/profile", style={"margin": "0 15px", "color": "white", "textDecoration": "none"})
    ], style={
        "backgroundColor": "#333",
        "padding": "10px",
        "display": "flex",
        "justifyContent": "center"
    })

# -------------------------
# Define the Index Page (Landing + Graph Section)
# -------------------------
def index_page():
    return html.Div([
        # Landing Page Section with Background Image
        html.Div(
            id="first-part",
            style={"position": "relative", "height": "100vh"},
            children=[
                html.Div(className="landing-background"),
                html.Div(
                    className="content",
                    children=[
                        html.H1("Child Penalty Explorer: Insights from Germany", className="main-heading"),
                        html.H3("Analyzing how parenthood affects wages, working hours, and employment opportunities", className="sub-heading"),
                        html.Div(
                            html.A("Explore Now", href="#second-part", className="explore-button"),
                            className="button-container"
                        ),
                        html.Div("Created by Merve Ogretmek", className="footer")
                    ]
                )
            ]
        ),
        # Second Section: Graph & Dropdown (with its own solid background)
        html.Div(
            id="second-part",
            style={"backgroundColor": "#f0f0f0", "paddingBottom": "50px"},
            children=[
                # Navigation Bar
                navbar(),
                html.Div(
                    children=[
                        html.P(
                            "The “child penalty” refers to the long-term impact on mother’s wages, hours, career trajectories after having a child, "
                            "relative to fathers. This phenomenon not only carries significant economic implications but also highlights persistent gender "
                            "inequalities in the labor market, emphasizing its relevance for shaping effective public policy."
                        ),
                        html.P(
                            "The analysis draws on data from the German Socio-Economic Panel (SOEP) covering the years 1984–2020 and employs an event-study approach "
                            "to trace these labor market dynamics over time. For more details on our methodology, please visit the dedicated “Methodology” page."
                        ),
                        html.H2("Big Picture in Germany", style={"textAlign": "center", "marginTop": "40px"}),
                        html.Div([
                            dcc.Graph(
                                id="child-penalty-graph", 
                                style={"width": "100%", "maxWidth": "800px", "margin": "0 auto"}
                            ),
                            dcc.Dropdown(
                                id="measure-dropdown",
                                options=[
                                    {"label": "Real Labor Income", "value": "ia"},
                                    {"label": "Working Hours (Annual)", "value": "wha"},
                                    {"label": "Employment Status", "value": "esa"}
                                ],
                                value="ia",
                                clearable=False,
                                style={"width": "50%", "margin": "20px auto"}
                            )
                        ], style={"textAlign": "center"}),
                        html.Div(
                            html.A("Explore Subgroups", href="/heterogeneous", className="explore-button"),
                            style={"marginTop": "30px", "textAlign": "center"}
                        )
                    ],
                    style={"padding": "50px", "maxWidth": "900px", "margin": "0 auto"}
                )
            ]
        )
    ])

# -------------------------
# Define the Methodology Page
# -------------------------
def methodology_page():
    return html.Div([
        navbar(),
        html.Div([
            html.H1("Methodology", style={
                "textAlign": "center", 
                "marginTop": "30px", 
                "color": "#333", 
                "fontWeight": "bold", 
                "fontSize": "36px"
            }),
            # Research Aims & Questions Section
            html.Div([
                html.H2("Research Aims & Questions", style={
                    "color": "#444", 
                    "borderBottom": "2px solid #007BFF", 
                    "paddingBottom": "10px",
                    "marginTop": "30px"
                }),
                html.Ul([
                    html.Li("How does having a child change women’s earnings and job participation?", style={"margin": "10px 0"}),
                    html.Li("Which groups face a higher or lower penalty?", style={"margin": "10px 0"}),
                    html.Li("Does the 2007 Parental Leave Reform (Elterngeld) reduce these penalties?", style={"margin": "10px 0"})
                ], style={
                    "paddingLeft": "20px", 
                    "lineHeight": "1.6", 
                    "fontSize": "16px", 
                    "color": "#555"
                })
            ]),
            # Sample Criteria Section
            html.Div([
                html.H2(["Sample Criteria", html.Sup("1")], style={
                    "color": "#444", 
                    "borderBottom": "2px solid #007BFF", 
                    "paddingBottom": "10px",
                    "marginTop": "30px"
                }),
                html.Ul([
                    html.Li("Individuals aged 20-45 at first childbirth", style={"margin": "10px 0"}),
                    html.Li("Excluding income from secondary jobs and self-employment", style={"margin": "10px 0"}),
                    html.Li("Observed at least once before and once after childbirth", style={"margin": "10px 0"}),
                    html.Li("At least 8 total observations", style={"margin": "10px 0"})
                ], style={
                    "paddingLeft": "20px", 
                    "lineHeight": "1.6", 
                    "fontSize": "16px", 
                    "color": "#555"
                })
            ]),
            # Event-Study Design Section
            html.Div([
                html.H2("Event-Study Design", style={
                    "color": "#444", 
                    "borderBottom": "2px solid #007BFF", 
                    "paddingBottom": "10px",
                    "marginTop": "30px"
                }),
                html.Div([
                    html.P([html.B("Core Idea:"), 
                            " Compare women’s career trajectories around the time of first childbirth to a “no-birth” baseline, controlling for age and year."],
                           style={"fontSize": "16px", "color": "#555", "margin": "10px 0"}),
                    html.P(html.B("Timeline:"), style={"fontSize": "16px", "color": "#555", "margin": "10px 0"}),
                    html.Ul([
                        html.Li("Define year of birth as Event Time 0", style={"margin": "8px 0"}),
                        html.Li("Negative years (-1, -2, ...) refer to pre-birth, positives (+1, +2, ...) to post-birth", style={"margin": "8px 0"})
                    ], style={
                        "paddingLeft": "20px", 
                        "lineHeight": "1.6", 
                        "fontSize": "16px", 
                        "color": "#555"
                    }),
                    html.P(html.B("What We Estimate:"), style={"fontSize": "16px", "color": "#555", "margin": "10px 0"}),
                    html.Ul([
                        html.Li("Mothers: How income/hours/employment deviate from their own pre-birth levels", style={"margin": "8px 0"}),
                        html.Li("Fathers: Provide a benchmark, to see how men’s outcomes evolve around fatherhood", style={"margin": "8px 0"}),
                        html.Li("Child Penalty: The gap between mothers’ post-birth changes and fathers’ post-birth changes, expressed as a percentage of a counterfactual (no-birth) trajectory", style={"margin": "8px 0"})
                    ], style={
                        "paddingLeft": "20px", 
                        "lineHeight": "1.6", 
                        "fontSize": "16px", 
                        "color": "#555"
                    })
                ])
            ])
        ], style={"maxWidth": "900px", "margin": "0 auto", "padding": "20px"}),
        # Footnote Section at the bottom
        html.Div([
            html.P("1. These criteria are adopted from Kleven et al. (2019)", style={
                "fontSize": "14px", 
                "fontStyle": "italic", 
                "color": "#777",
                "textAlign": "center",
                "marginTop": "40px"
            })
        ])
    ])

# -------------------------
# Define the Heterogeneous Analysis Page
# -------------------------
# Define measure and subgroup labels (exact order as required)
measure_labels = {
    "ia": "Real Labor Income",
    "wha": "Working Hours (Annual)",
    "esa": "Employment Status"
}

subgroup_labels = {
    # Residential Area
    "rural": "Rural Residents",
    "urban": "Urban Residents",
    # Geographical Area
    "east": "Parents in East Germany",
    "west": "Parents in West Germany",
    # Sectors
    "malesector": "Male-Dominated Sectors",
    "femalesector": "Female-Dominated Sectors",
    "balancedsector": "Balanced-Gender Sectors",
    # Origin
    "immigrant": "Immigrants",
    "native": "Natives",
    # Education Level
    "lowedu": "Parents with Low-Education",
    "highedu": "Parents with High-Education",
    # Partnership Status
    "nopartner": "Parents without a Partner",
    "partnered": "Parents with a Partner",
    # Age of Parenthood
    "young": "Early Parents",
    "median": "Median Parents",
    "old": "Late Parents"
}

measure_options = [{"label": measure_labels[m], "value": m} for m in ["ia", "wha", "esa"]]

subgroup_options = [
    {"label": "Residential Area", "value": "Residential Area", "disabled": True},
    {"label": "   " + subgroup_labels["rural"], "value": "rural"},
    {"label": "   " + subgroup_labels["urban"], "value": "urban"},
    {"label": "Geographical Area", "value": "Geographical Area", "disabled": True},
    {"label": "   " + subgroup_labels["east"], "value": "east"},
    {"label": "   " + subgroup_labels["west"], "value": "west"},
    {"label": "Sectors", "value": "Sectors", "disabled": True},
    {"label": "   " + subgroup_labels["malesector"], "value": "malesector"},
    {"label": "   " + subgroup_labels["femalesector"], "value": "femalesector"},
    {"label": "   " + subgroup_labels["balancedsector"], "value": "balancedsector"},
    {"label": "Origin", "value": "Origin", "disabled": True},
    {"label": "   " + subgroup_labels["immigrant"], "value": "immigrant"},
    {"label": "   " + subgroup_labels["native"], "value": "native"},
    {"label": "Education Level", "value": "Education Level", "disabled": True},
    {"label": "   " + subgroup_labels["lowedu"], "value": "lowedu"},
    {"label": "   " + subgroup_labels["highedu"], "value": "highedu"},
    {"label": "Partnership Status", "value": "Partnership Status", "disabled": True},
    {"label": "   " + subgroup_labels["nopartner"], "value": "nopartner"},
    {"label": "   " + subgroup_labels["partnered"], "value": "partnered"},
    {"label": "Age of Parenthood", "value": "Age of Parenthood", "disabled": True},
    {"label": "   " + subgroup_labels["young"], "value": "young"},
    {"label": "   " + subgroup_labels["median"], "value": "median"},
    {"label": "   " + subgroup_labels["old"], "value": "old"}
]

def heterogeneous_analysis_page():
    return html.Div([
        navbar(),
        html.Div([
            html.H1("Heterogeneous Analysis", style={
                "textAlign": "center", 
                "marginTop": "30px", 
                "color": "#333",
                "fontWeight": "bold", 
                "fontSize": "36px"
            }),
            html.Div([
                html.Label("Select Measure:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="het-measure-dropdown",
                    options=measure_options,
                    value="ia",
                    clearable=False,
                    style={"width": "300px", "display": "inline-block"}
                )
            ], style={"textAlign": "center", "marginTop": "20px"}),
            html.Div([
                html.Label("Select Subgroup:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="het-subgroup-dropdown",
                    options=subgroup_options,
                    value="rural",
                    clearable=False,
                    style={"width": "300px", "display": "inline-block"}
                )
            ], style={"textAlign": "center", "marginTop": "20px"}),
            html.Div([
                dcc.Graph(
                    id="heterogeneous-graph",
                    style={"width": "100%", "maxWidth": "800px", "margin": "30px auto"}
                )
            ])
        ], style={"maxWidth": "900px", "margin": "0 auto", "padding": "20px"})
    ])

# -------------------------
# Define the Reform Analysis Page
# -------------------------
# Full dataset from your LaTeX table:
data_reform = [
    # --- GENERAL (6 rows) ---
    {"Category": "General", "Subcategory": "-", "Approach": "Income Approach",              "Period": "Before 2007", "Total": 42389, "Male": 20453, "Female": 21936, "LR_Child_Penalty": 0.52},
    {"Category": "General", "Subcategory": "-", "Approach": "Income Approach",              "Period": "After 2007",  "Total": 15441, "Male": 6820,  "Female": 8621,  "LR_Child_Penalty": 0.26},
    {"Category": "General", "Subcategory": "-", "Approach": "Working Hours Approach",       "Period": "Before 2007", "Total": 43600, "Male": 21183, "Female": 22417, "LR_Child_Penalty": 0.66},
    {"Category": "General", "Subcategory": "-", "Approach": "Working Hours Approach",       "Period": "After 2007",  "Total": 16379, "Male": 7414,  "Female": 8965,  "LR_Child_Penalty": 0.32},
    {"Category": "General", "Subcategory": "-", "Approach": "Employment Status Approach",   "Period": "Before 2007", "Total": 43481, "Male": 21130, "Female": 22351, "LR_Child_Penalty": 0.44},
    {"Category": "General", "Subcategory": "-", "Approach": "Employment Status Approach",   "Period": "After 2007",  "Total": 16379, "Male": 7414,  "Female": 8965,  "LR_Child_Penalty": 0.07},

    # --- RURAL vs URBAN (12 rows) ---
    # Rural
    {"Category": "Rural vs Urban", "Subcategory": "Rural Residential Area", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 21075, "Male": 10296, "Female": 10779, "LR_Child_Penalty": 0.59},
    {"Category": "Rural vs Urban", "Subcategory": "Rural Residential Area", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 6277,  "Male": 2826,  "Female": 3451,  "LR_Child_Penalty": 0.40},
    {"Category": "Rural vs Urban", "Subcategory": "Rural Residential Area", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 21739, "Male": 10694, "Female": 11045, "LR_Child_Penalty": 0.72},
    {"Category": "Rural vs Urban", "Subcategory": "Rural Residential Area", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 6653,  "Male": 3078,  "Female": 3575,  "LR_Child_Penalty": 0.35},
    {"Category": "Rural vs Urban", "Subcategory": "Rural Residential Area", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 21695, "Male": 10675, "Female": 11020, "LR_Child_Penalty": 0.42},
    {"Category": "Rural vs Urban", "Subcategory": "Rural Residential Area", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 6653,  "Male": 3078,  "Female": 3575,  "LR_Child_Penalty": 0.00001},
    # Urban
    {"Category": "Rural vs Urban", "Subcategory": "Urban Residential Area", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 21314, "Male": 10157, "Female": 11157, "LR_Child_Penalty": 0.45},
    {"Category": "Rural vs Urban", "Subcategory": "Urban Residential Area", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 9164,  "Male": 3994,  "Female": 5170,  "LR_Child_Penalty": 0.20},
    {"Category": "Rural vs Urban", "Subcategory": "Urban Residential Area", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 21861, "Male": 10489, "Female": 11372, "LR_Child_Penalty": 0.60},
    {"Category": "Rural vs Urban", "Subcategory": "Urban Residential Area", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 9726,  "Male": 4336,  "Female": 5390,  "LR_Child_Penalty": 0.30},
    {"Category": "Rural vs Urban", "Subcategory": "Urban Residential Area", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 21786, "Male": 10455, "Female": 11331, "LR_Child_Penalty": 0.45},
    {"Category": "Rural vs Urban", "Subcategory": "Urban Residential Area", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 9726,  "Male": 4336,  "Female": 5390,  "LR_Child_Penalty": 0.12},

    # --- WEST vs EAST (12 rows) ---
    # West Germany
    {"Category": "West vs East", "Subcategory": "West Germany", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 34713, "Male": 16883, "Female": 17830, "LR_Child_Penalty": 0.58},
    {"Category": "West vs East", "Subcategory": "West Germany", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 11521, "Male": 5246,  "Female": 6275,  "LR_Child_Penalty": 0.30},
    {"Category": "West vs East", "Subcategory": "West Germany", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 35753, "Male": 17497, "Female": 18256, "LR_Child_Penalty": 0.72},
    {"Category": "West vs East", "Subcategory": "West Germany", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 12224, "Male": 5682,  "Female": 6542,  "LR_Child_Penalty": 0.35},
    {"Category": "West vs East", "Subcategory": "West Germany", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 35753, "Male": 17497, "Female": 18256, "LR_Child_Penalty": 0.46},
    {"Category": "West vs East", "Subcategory": "West Germany", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 12224, "Male": 5682,  "Female": 6542,  "LR_Child_Penalty": 0.03},
    # East Germany
    {"Category": "West vs East", "Subcategory": "East Germany", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 7676,  "Male": 3570,  "Female": 4106,  "LR_Child_Penalty": 0.29},
    {"Category": "West vs East", "Subcategory": "East Germany", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 3920,  "Male": 1574,  "Female": 2346,  "LR_Child_Penalty": 0.30},
    {"Category": "West vs East", "Subcategory": "East Germany", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 7847,  "Male": 3686,  "Female": 4161,  "LR_Child_Penalty": 0.42},
    {"Category": "West vs East", "Subcategory": "East Germany", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 4155,  "Male": 1732,  "Female": 2423,  "LR_Child_Penalty": 0.30},
    {"Category": "West vs East", "Subcategory": "East Germany", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 7728,  "Male": 3633,  "Female": 4095,  "LR_Child_Penalty": 0.35},
    {"Category": "West vs East", "Subcategory": "East Germany", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 4155,  "Male": 1732,  "Female": 2423,  "LR_Child_Penalty": 0.27},

    # --- SECTOR GENDER RATIO (18 rows) ---
    # Male-Dominated
    {"Category": "Sector Gender Ratio", "Subcategory": "Male-Dominated Sectors", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 8106, "Male": 6390, "Female": 1716, "LR_Child_Penalty": 0.35},
    {"Category": "Sector Gender Ratio", "Subcategory": "Male-Dominated Sectors", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 2525, "Male": 1874, "Female": 651,  "LR_Child_Penalty": 0.37},
    {"Category": "Sector Gender Ratio", "Subcategory": "Male-Dominated Sectors", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 8171, "Male": 6451, "Female": 1720, "LR_Child_Penalty": 0.32},
    {"Category": "Sector Gender Ratio", "Subcategory": "Male-Dominated Sectors", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 2543, "Male": 1890, "Female": 653,  "LR_Child_Penalty": 0.42},
    {"Category": "Sector Gender Ratio", "Subcategory": "Male-Dominated Sectors", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 8124, "Male": 6418, "Female": 1706, "LR_Child_Penalty": 0.04},
    {"Category": "Sector Gender Ratio", "Subcategory": "Male-Dominated Sectors", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 2543, "Male": 1890, "Female": 653,  "LR_Child_Penalty": -0.05},
    # Female-Dominated
    {"Category": "Sector Gender Ratio", "Subcategory": "Female-Dominated Sectors", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 4870, "Male": 1264, "Female": 3606, "LR_Child_Penalty": 0.29},
    {"Category": "Sector Gender Ratio", "Subcategory": "Female-Dominated Sectors", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 2165, "Male": 554,  "Female": 1611, "LR_Child_Penalty": 0.62},
    {"Category": "Sector Gender Ratio", "Subcategory": "Female-Dominated Sectors", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 4898, "Male": 1270, "Female": 3628, "LR_Child_Penalty": 0.53},
    {"Category": "Sector Gender Ratio", "Subcategory": "Female-Dominated Sectors", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 2179, "Male": 559,  "Female": 1620, "LR_Child_Penalty": 0.17},
    {"Category": "Sector Gender Ratio", "Subcategory": "Female-Dominated Sectors", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 4869, "Male": 1263, "Female": 3606, "LR_Child_Penalty": 0.0001},
    {"Category": "Sector Gender Ratio", "Subcategory": "Female-Dominated Sectors", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 2179, "Male": 559,  "Female": 1620, "LR_Child_Penalty": 0.000001},
    # Balanced
    {"Category": "Sector Gender Ratio", "Subcategory": "Balanced Gender Sectors", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 28987, "Male": 12600, "Female": 16387, "LR_Child_Penalty": 0.54},
    {"Category": "Sector Gender Ratio", "Subcategory": "Balanced Gender Sectors", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 10520, "Male": 4280,  "Female": 6240,  "LR_Child_Penalty": 0.17},
    {"Category": "Sector Gender Ratio", "Subcategory": "Balanced Gender Sectors", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 29167, "Male": 12700, "Female": 16467, "LR_Child_Penalty": 0.65},
    {"Category": "Sector Gender Ratio", "Subcategory": "Balanced Gender Sectors", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 10656, "Male": 4368,  "Female": 6288,  "LR_Child_Penalty": 0.37},
    {"Category": "Sector Gender Ratio", "Subcategory": "Balanced Gender Sectors", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 29129, "Male": 12689, "Female": 16440, "LR_Child_Penalty": 0.47},
    {"Category": "Sector Gender Ratio", "Subcategory": "Balanced Gender Sectors", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 10656, "Male": 4368,  "Female": 6288,  "LR_Child_Penalty": 0.25},

    # --- ORIGIN (12 rows) ---
    # Native-Born
    {"Category": "Origin", "Subcategory": "Native-Born", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 35110, "Male": 16543, "Female": 18567, "LR_Child_Penalty": 0.48},
    {"Category": "Origin", "Subcategory": "Native-Born", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 14137, "Male": 6248,  "Female": 7889,  "LR_Child_Penalty": 0.26},
    {"Category": "Origin", "Subcategory": "Native-Born", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 35332, "Male": 16684, "Female": 18648, "LR_Child_Penalty": 0.61},
    {"Category": "Origin", "Subcategory": "Native-Born", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 14290, "Male": 6345,  "Female": 7945,  "LR_Child_Penalty": 0.42},
    {"Category": "Origin", "Subcategory": "Native-Born", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 35219, "Male": 16634, "Female": 18585, "LR_Child_Penalty": 0.34},
    {"Category": "Origin", "Subcategory": "Native-Born", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 14290, "Male": 6345,  "Female": 7945,  "LR_Child_Penalty": 0.19},
    # Immigrant
    {"Category": "Origin", "Subcategory": "Immigrant", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 6837, "Male": 3703, "Female": 3134, "LR_Child_Penalty": 0.88},
    {"Category": "Origin", "Subcategory": "Immigrant", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 1103, "Male": 471,  "Female": 632,  "LR_Child_Penalty": 0.26},
    {"Category": "Origin", "Subcategory": "Immigrant", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 6887, "Male": 3729, "Female": 3158, "LR_Child_Penalty": 0.76},
    {"Category": "Origin", "Subcategory": "Immigrant", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 1119, "Male": 483,  "Female": 636,  "LR_Child_Penalty": 0.93},
    {"Category": "Origin", "Subcategory": "Immigrant", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 6886, "Male": 3728, "Female": 3158, "LR_Child_Penalty": 0.64},
    {"Category": "Origin", "Subcategory": "Immigrant", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 1119, "Male": 483,  "Female": 636,  "LR_Child_Penalty": 0.79},

    # --- EDUCATION LEVEL (12 rows) ---
    # Low Education
    {"Category": "Education Level", "Subcategory": "Low Education", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 24747, "Male": 12262, "Female": 12485, "LR_Child_Penalty": 0.67},
    {"Category": "Education Level", "Subcategory": "Low Education", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 5300,  "Male": 2513,  "Female": 2787,  "LR_Child_Penalty": 0.30},
    {"Category": "Education Level", "Subcategory": "Low Education", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 24892, "Male": 12358, "Female": 12534, "LR_Child_Penalty": 0.61},
    {"Category": "Education Level", "Subcategory": "Low Education", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 5376,  "Male": 2570,  "Female": 2806,  "LR_Child_Penalty": 0.36},
    {"Category": "Education Level", "Subcategory": "Low Education", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 24804, "Male": 12317, "Female": 12487, "LR_Child_Penalty": 0.42},
    {"Category": "Education Level", "Subcategory": "Low Education", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 5376,  "Male": 2570,  "Female": 2806,  "LR_Child_Penalty": 0.20},
    # High Education
    {"Category": "Education Level", "Subcategory": "High Education", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 15946, "Male": 7467,  "Female": 8479,  "LR_Child_Penalty": 0.43},
    {"Category": "Education Level", "Subcategory": "High Education", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 8968,  "Male": 3892,  "Female": 5076,  "LR_Child_Penalty": 0.30},
    {"Category": "Education Level", "Subcategory": "High Education", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 16058, "Male": 7530,  "Female": 8528,  "LR_Child_Penalty": 0.60},
    {"Category": "Education Level", "Subcategory": "High Education", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 9041,  "Male": 3930,  "Female": 5111,  "LR_Child_Penalty": 0.45},
    {"Category": "Education Level", "Subcategory": "High Education", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 16032, "Male": 7520,  "Female": 8512,  "LR_Child_Penalty": 0.30},
    {"Category": "Education Level", "Subcategory": "High Education", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 9041,  "Male": 3930,  "Female": 5111,  "LR_Child_Penalty": 0.20},

    # --- PARENTHOOD AGE (18 rows) ---
    # Early Parent
    {"Category": "Parenthood Age", "Subcategory": "Early Parent", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 9070, "Male": 2885, "Female": 6185, "LR_Child_Penalty": 0.52},
    {"Category": "Parenthood Age", "Subcategory": "Early Parent", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 1551, "Male": 402,  "Female": 1149, "LR_Child_Penalty": 0.36},
    {"Category": "Parenthood Age", "Subcategory": "Early Parent", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 9290, "Male": 2960, "Female": 6330, "LR_Child_Penalty": 0.61},
    {"Category": "Parenthood Age", "Subcategory": "Early Parent", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 1622, "Male": 445,  "Female": 1177, "LR_Child_Penalty": 0.34},
    {"Category": "Parenthood Age", "Subcategory": "Early Parent", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 9242, "Male": 2946, "Female": 6296, "LR_Child_Penalty": 0.52},
    {"Category": "Parenthood Age", "Subcategory": "Early Parent", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 1622, "Male": 445,  "Female": 1177, "LR_Child_Penalty": -0.04},
    # Median Parent
    {"Category": "Parenthood Age", "Subcategory": "Median Parent", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 28690, "Male": 14434, "Female": 14256, "LR_Child_Penalty": 0.47},
    {"Category": "Parenthood Age", "Subcategory": "Median Parent", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 9279,  "Male": 3702,  "Female": 5577,  "LR_Child_Penalty": 0.26},
    {"Category": "Parenthood Age", "Subcategory": "Median Parent", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 29438, "Male": 14883, "Female": 14555, "LR_Child_Penalty": 0.70},
    {"Category": "Parenthood Age", "Subcategory": "Median Parent", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 9872,  "Male": 4082,  "Female": 5790,  "LR_Child_Penalty": 0.57},
    {"Category": "Parenthood Age", "Subcategory": "Median Parent", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 29370, "Male": 14845, "Female": 14525, "LR_Child_Penalty": 0.41},
    {"Category": "Parenthood Age", "Subcategory": "Median Parent", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 9872,  "Male": 4082,  "Female": 5790,  "LR_Child_Penalty": 0.21},
    # Late Parent
    {"Category": "Parenthood Age", "Subcategory": "Late Parent", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 4629, "Male": 3134, "Female": 1495, "LR_Child_Penalty": 0.11},
    {"Category": "Parenthood Age", "Subcategory": "Late Parent", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 4611, "Male": 2716, "Female": 1895, "LR_Child_Penalty": 0.08},
    {"Category": "Parenthood Age", "Subcategory": "Late Parent", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 4872, "Male": 3340, "Female": 1532, "LR_Child_Penalty": 0.93},
    {"Category": "Parenthood Age", "Subcategory": "Late Parent", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 4885, "Male": 2887, "Female": 1998, "LR_Child_Penalty": 0.00001},
    {"Category": "Parenthood Age", "Subcategory": "Late Parent", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 4869, "Male": 3339, "Female": 1530, "LR_Child_Penalty": 0.45},
    {"Category": "Parenthood Age", "Subcategory": "Late Parent", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 4885, "Male": 2887, "Female": 1998, "LR_Child_Penalty": -0.31},

    # --- PARTNERSHIP STATUS (12 rows) ---
    # Partnered
    {"Category": "Partnership Status", "Subcategory": "Partnered", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 29631, "Male": 14634, "Female": 14997, "LR_Child_Penalty": 0.47},
    {"Category": "Partnership Status", "Subcategory": "Partnered", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 8843,  "Male": 4086,  "Female": 4757,  "LR_Child_Penalty": 0.22},
    {"Category": "Partnership Status", "Subcategory": "Partnered", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 29816, "Male": 14748, "Female": 15068, "LR_Child_Penalty": 0.55},
    {"Category": "Partnership Status", "Subcategory": "Partnered", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 8953,  "Male": 4163,  "Female": 4790,  "LR_Child_Penalty": 0.43},
    {"Category": "Partnership Status", "Subcategory": "Partnered", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 29776, "Male": 14728, "Female": 15048, "LR_Child_Penalty": 0.40},
    {"Category": "Partnership Status", "Subcategory": "Partnered", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 8953,  "Male": 4163,  "Female": 4790,  "LR_Child_Penalty": 0.17},
    # No-Partner
    {"Category": "Partnership Status", "Subcategory": "No-Partner", "Approach": "Income Approach",            "Period": "Before 2007", "Total": 12265, "Male": 5585,  "Female": 6680,  "LR_Child_Penalty": 0.23},
    {"Category": "Partnership Status", "Subcategory": "No-Partner", "Approach": "Income Approach",            "Period": "After 2007",  "Total": 6361,  "Male": 2613,  "Female": 3748,  "LR_Child_Penalty": 0.14},
    {"Category": "Partnership Status", "Subcategory": "No-Partner", "Approach": "Working Hours Approach",     "Period": "Before 2007", "Total": 12351, "Male": 5637,  "Female": 6714,  "LR_Child_Penalty": 0.29},
    {"Category": "Partnership Status", "Subcategory": "No-Partner", "Approach": "Working Hours Approach",     "Period": "After 2007",  "Total": 6420,  "Male": 2645,  "Female": 3775,  "LR_Child_Penalty": 0.18},
    {"Category": "Partnership Status", "Subcategory": "No-Partner", "Approach": "Employment Status Approach", "Period": "Before 2007", "Total": 12277, "Male": 5606,  "Female": 6671,  "LR_Child_Penalty": 0.20},
    {"Category": "Partnership Status", "Subcategory": "No-Partner", "Approach": "Employment Status Approach", "Period": "After 2007",  "Total": 6420,  "Male": 2645,  "Female": 3775,  "LR_Child_Penalty": 0.02},
]

df_reform = pd.DataFrame(data_reform)

def reform_analysis_page():
    return html.Div([
        navbar(),
        html.Div([
            html.H1("Reform Analysis: Long-Run Child Penalty Summary", 
                    style={"textAlign": "center", "marginTop": "30px", "color": "#333"}),
            html.Div([
                html.Label("Select Category:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="reform-category-dropdown",
                    options=[{"label": cat, "value": cat} for cat in df_reform["Category"].unique()],
                    value="General",
                    clearable=False,
                    style={"width": "300px"}
                )
            ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "padding": "20px"}),
            html.Div([
                # Just the graph now, no summary Div
                dcc.Graph(id="reform-bar-chart")
            ], style={"width": "65%", "display": "inline-block", "paddingLeft": "5%"})
        ], style={"maxWidth": "900px", "margin": "0 auto", "padding": "20px"})
    ])

# -------------------------
# Define the Build Your Profile Page
# -------------------------
def build_profile_page():
    return html.Div([
        navbar(),
        html.H1("Build Your Profile and Predict Long-Run Effect of First Childbirth", 
                style={"textAlign": "center", "marginTop": "30px"}),
        # Build Profile Form
        html.Div([
            html.Label("Measurement Type:", className="build-profile-label"),
            dcc.Dropdown(
                id='outcome-type',
                options=[
                    {'label': 'Real Labor Income', 'value': 'ia'},
                    {'label': 'Working Hours (Annual)', 'value': 'wha'},
                    {'label': 'Employment Status', 'value': 'esa'},
                ],
                value='ia',
                className="build-profile-dropdown"
            ),
            html.Label("Gender:", className="build-profile-label"),
            dcc.RadioItems(
                id='gender',
                options=[
                    {'label': 'Male', 'value': 'male'},
                    {'label': 'Female', 'value': 'female'}
                ],
                value='male',
                labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                className="build-profile-radio"
            ),
            html.Label("Residential Area:", className="build-profile-label"),
            dcc.RadioItems(
                id='area-type',
                options=[
                    {'label': 'Rural', 'value': 'rural'},
                    {'label': 'Urban', 'value': 'urban'}
                ],
                value='rural',
                labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                className="build-profile-radio"
            ),
            html.Label("Region:", className="build-profile-label"),
            dcc.RadioItems(
                id='region-type',
                options=[
                    {'label': 'West Germany', 'value': 'west'},
                    {'label': 'East Germany', 'value': 'east'}
                ],
                value='west',
                labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                className="build-profile-radio"
            ),
            html.Label("Education Level:", className="build-profile-label"),
            dcc.RadioItems(
                id='edu-type',
                options=[
                    {'label': 'Low Education (until high school degree)', 'value': 'lowedu'},
                    {'label': 'High Education (high school degree and after)', 'value': 'highedu'}
                ],
                value='lowedu',
                labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                className="build-profile-radio"
            ),
            html.Label("Partner Status:", className="build-profile-label"),
            dcc.RadioItems(
                id='partner-status',
                options=[
                    {'label': 'Without Partner', 'value': 'nopartner'},
                    {'label': 'With Partner', 'value': 'partnered'}
                ],
                value='nopartner',
                labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                className="build-profile-radio"
            ),
            html.Label("Origin:", className="build-profile-label"),
            dcc.RadioItems(
                id='nativity-type',
                options=[
                    {'label': 'Native', 'value': 'native'},
                    {'label': 'Immigrant', 'value': 'immigrant'}
                ],
                value='native',
                labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                className="build-profile-radio"
            ),
            html.Label("Age at First Parenthood:", className="build-profile-label"),
            dcc.RadioItems(
                id='age-group',
                options=[
                    {'label': 'Young (<25)', 'value': 'young'},
                    {'label': 'Median (25-34)', 'value': 'median'},
                    {'label': 'Old (>34)', 'value': 'old'}
                ],
                value='median',
                labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                className="build-profile-radio"
            ),
            html.Button('Build Profile', id='build-profile-btn', n_clicks=0, className="build-profile-button")
        ], className="build-profile-form"),
        html.Hr(),
        # Predicted Penalty Output
        html.Div(id='profile-output', className="profile-output"),
        # Explanation Section for the Predicted Long-Run Child Penalty
        html.Div([
            html.H3("What Does the Predicted Long-Run Effect Mean?"),
            dcc.Markdown('''\
The predicted long-run effect represents the estimated long-term impact that becoming a parent has on your labor market outcomes. In practical terms, it shows by what percentage your expected outcomes—such as earnings, working hours, or employment status—might be lower compared to a scenario where you did not have a child. For instance, if the penalty is **0.30**, it suggests that your outcomes could be 30% lower in the long run due to childbearing. This number summarizes the cumulative effect of parenthood on your career trajectory.
            ''')
        ], style={"maxWidth": "900px", "margin": "0 auto", "padding": "20px"})
    ])

# -------------------------
# Main Layout with dcc.Location for Multi-Page Routing
# -------------------------
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

# -------------------------
# Callback for Multi-Page Routing
# -------------------------
@app.callback(Output("page-content", "children"),
              [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/methodology":
        return methodology_page()
    elif pathname == "/heterogeneous":
        return heterogeneous_analysis_page()
    elif pathname == "/reform":
        return reform_analysis_page()
    elif pathname == "/profile":
        return build_profile_page()
    else:
        return index_page()

# -------------------------
# Callback to Update the Graph on the Index Page
# -------------------------
@app.callback(
    Output("child-penalty-graph", "figure"),
    Input("measure-dropdown", "value")
)
def update_graph(selected_measure):
    file_map = {
        "ia": "results/general_results_ia.csv",
        "wha": "results/general_results_wha.csv",
        "esa": "results/general_results_esa.csv"
    }
    file_name = file_map.get(selected_measure, "results/general_results_ia.csv")
    try:
        df = pd.read_csv(file_name)
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title="Data not available. Please check your CSV files.")
        return fig
    mean_m = df.loc[df['event_time'] >= 0, 'percentage_coef_m'].mean() - df.loc[df['event_time'] < 0, 'percentage_coef_m'].mean()
    mean_w = df.loc[df['event_time'] >= 0, 'percentage_coef_w'].mean() - df.loc[df['event_time'] < 0, 'percentage_coef_w'].mean()
    child_penalty = mean_m - mean_w
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['event_time'],
        y=df['percentage_coef_m'],
        mode='lines+markers',
        name='Male',
        line=dict(color='black', dash='dash'),
        marker=dict(symbol='circle')
    ))
    fig.add_trace(go.Scatter(
        x=df['event_time'],
        y=df['percentage_coef_w'],
        mode='lines+markers',
        name='Female',
        line=dict(color='black'),
        marker=dict(symbol='triangle-up', size=10)
    ))
    # Add vertical dashed gray line at x = -0.5 (in between event_time -1 and 0)
    fig.add_shape(
        type="line",
        x0=-0.5, x1=-0.5,
        y0=-1, y1=0.5,
        xref="x", yref="y",
        line=dict(color="gray", dash="dash")
    )
    annotation_text = f"Child Penalty: {child_penalty:.2f}"
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=10, color="black")
    )
    fig.update_layout(
        height=350,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            title="Event Time (Birth of the First Child)",
            showline=True,
            linecolor="black",
            mirror=False,
            ticks="outside",
            tickmode="linear",
            dtick=1,
            showgrid=False
        ),
        yaxis=dict(
            title="Relative Effect",
            range=[-1, 0.5],
            tickvals=[-0.5, 0, 0.5],
            showline=True,
            linecolor="black",
            mirror=False,
            ticks="outside",
            showgrid=True,
            gridcolor="#d3d3d3",
            gridwidth=1,
            griddash="dash"
        ),
        showlegend=True
    )
    return fig

# -------------------------
# Callback to Update the Heterogeneous Analysis Graph
# -------------------------
@app.callback(
    Output("heterogeneous-graph", "figure"),
    [Input("het-measure-dropdown", "value"),
     Input("het-subgroup-dropdown", "value")]
)
def update_heterogeneous_graph(selected_measure, selected_subgroup):
    file_path = f"results/heterogeneous_analysis/{selected_subgroup}_results_{selected_measure}.csv"
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            title="Data not available. Please check your CSV files.",
            height=350,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        return fig
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['event_time'],
        y=df['percentage_coef_m'],
        mode='lines+markers',
        name='Male',
        line=dict(color='black', dash='dash'),
        marker=dict(symbol='circle')
    ))
    fig.add_trace(go.Scatter(
        x=df['event_time'],
        y=df['percentage_coef_w'],
        mode='lines+markers',
        name='Female',
        line=dict(color='black'),
        marker=dict(symbol='triangle-up', size=10)
    ))
    # Add vertical dashed gray line at x = -0.5
    fig.add_shape(
        type="line",
        x0=-0.5, x1=-0.5,
        y0=-1, y1=0.5,
        xref="x", yref="y",
        line=dict(color="gray", dash="dash")
    )
    fig.update_layout(
        height=350,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            title="Event Time (Birth of the First Child)",
            showline=True,
            linecolor="black",
            mirror=False,
            ticks="outside",
            tickmode="linear",
            dtick=1,
            showgrid=False
        ),
        yaxis=dict(
            title="Relative Effect",
            range=[-1, 0.5],
            tickvals=[-0.5, 0, 0.5],
            showline=True,
            linecolor="black",
            mirror=False,
            ticks="outside",
            showgrid=True,
            gridcolor="#d3d3d3",
            gridwidth=1,
            griddash="dash"
        ),
        showlegend=True
    )
    return fig

# -------------------------
# Callback to Update the Reform Analysis Summary Page
# -------------------------
@app.callback(
    Output("reform-bar-chart", "figure"),
    Input("reform-category-dropdown", "value")
)
def update_reform_summary(selected_category):
    # Filter data for the selected category
    filtered_df = df_reform[df_reform["Category"] == selected_category]

    # Create a grouped bar chart with facets for subcategories
    fig = px.bar(
        filtered_df,
        x="Approach",
        y="LR_Child_Penalty",
        color="Period",
        barmode="group",
        facet_col="Subcategory",
        title=f"LR Child Penalty for {selected_category}",
        labels={"LR_Child_Penalty": "LR Child Penalty", "Approach": "Measure"},
        category_orders={
            "Period": ["Before 2007", "After 2007"],
            "Approach": [
                "Income Approach",
                "Working Hours Approach",
                "Employment Status Approach"
            ]
        },
        height=450,
        width=1000
    )

    # (Optional) Adjust axis ranges, etc.
    min_val = filtered_df["LR_Child_Penalty"].min()
    max_val = filtered_df["LR_Child_Penalty"].max()
    padding = 0.05
    fig.update_yaxes(range=[min_val - padding, max_val + padding], matches=None)
    fig.update_xaxes(matches=None)

    return fig

# -------------------------
# Callback to Build the Profile on the Build Your Profile Page
# -------------------------
@app.callback(
    Output('profile-output', 'children'),
    Input('build-profile-btn', 'n_clicks'),
    State('outcome-type', 'value'),
    State('gender', 'value'),
    State('area-type', 'value'),
    State('region-type', 'value'),
    State('edu-type', 'value'),
    State('partner-status', 'value'),
    State('nativity-type', 'value'),
    State('age-group', 'value')
)
def build_profile(n_clicks, outcome, gender, area, region, edu, partner, nativity, age):
    if n_clicks == 0:
        return ""
    
    # For the "gender" subgroup, map the user's input to the proper CSV file name.
    # There is no "male_results", so if the person is male, we load "malesector_results_{outcome}.csv"
    # and if the person is female, we load "femalesector_results_{outcome}.csv".
    gender_csv = "malesector" if gender == "male" else "femalesector"
    
    try:
        # Load CSV files for each subgroup
        area_df     = load_csv(f"{area}_results_{outcome}.csv")
        region_df   = load_csv(f"{region}_results_{outcome}.csv")
        gender_df   = load_csv(f"{gender_csv}_results_{outcome}.csv")  # Load the gender-specific CSV
        edu_df      = load_csv(f"{edu}_results_{outcome}.csv")
        partner_df  = load_csv(f"{partner}_results_{outcome}.csv")
        nativity_df = load_csv(f"{nativity}_results_{outcome}.csv")
        age_df      = load_csv(f"{age}_results_{outcome}.csv")
    except Exception as e:
        return html.Div(f"Error loading one or more CSV files: {str(e)}")
    
    # Function to calculate the long-run penalty for a given subgroup CSV.
    # If the person is male, use the column 'percentage_coef_m'.
    # If the person is female, use the column 'child_penalty' (which is relative to men).
    def calculate_long_run_penalty(df, is_male):
        # Filter for event_time 5 and after
        df_long_run = df[df['event_time'] >= 5]
        if df_long_run.empty:
            return None
        if is_male:
            return df_long_run['percentage_coef_m'].mean()
        else:
            return abs(df_long_run['percentage_coef_w'].mean())
    
    # Determine flag for calculation based on person's gender.
    is_male = (gender == "male")
    
    # Calculate long-run penalty for each subgroup
    area_penalty     = calculate_long_run_penalty(area_df, is_male)
    region_penalty   = calculate_long_run_penalty(region_df, is_male)
    gender_penalty   = calculate_long_run_penalty(gender_df, is_male)
    edu_penalty      = calculate_long_run_penalty(edu_df, is_male)
    partner_penalty  = calculate_long_run_penalty(partner_df, is_male)
    nativity_penalty = calculate_long_run_penalty(nativity_df, is_male)
    age_penalty      = calculate_long_run_penalty(age_df, is_male)
    
    # Aggregate the subgroup penalties by taking their average.
    penalties = [area_penalty, region_penalty, gender_penalty,
                 edu_penalty, partner_penalty, nativity_penalty, age_penalty]
    valid_penalties = [p for p in penalties if p is not None]
    if not valid_penalties:
        overall_penalty = None
    else:
        overall_penalty = sum(valid_penalties) / len(valid_penalties)
    
    # Present the predicted long-run child penalty
    result = html.Div([
        html.H3("Predicted Long-Run Child Penalty"),
        html.P(
            f"Overall predicted penalty: {overall_penalty:.2f}" 
            if overall_penalty is not None 
            else "No data available for event_time 5 and after."
        )
    ])
    
    return result

if __name__ == '__main__':
    app.run_server(debug=True)

