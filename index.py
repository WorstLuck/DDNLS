import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.subplots as subplots
import datetime
import requests
from dash.exceptions import PreventUpdate
import base64
import io
import dash_table

# # Using generic style sheet
external_stylesheets =[dbc.themes.SIMPLEX]

# Instantiate app and suppress callbacks
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True
server = app.server

input_style = {'textAlign':'center','width':'auto'}
result_style = {'textAlign':'center','fontWeight': 'bold','borderStyle': 'groove','borderColor': '#eeeeee','borderWidth': '1px'}
label_font = {'textAlign':'center'}

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([dbc.Label("Degrees of freedom N",style=label_font)],width={'size':2}),
        dbc.Col([dbc.Label("Disorder number β",style=label_font)],width={'size':2}),
        dbc.Col([dbc.Label("Disorder Strength W",style=label_font)],width={'size':2}),
        dbc.Col([dbc.Label("Normalisation parameter S",style=label_font)],width={'size':2}),
        dbc.Col([dbc.Label("Tolerance",style=label_font)],width={'size':2}),
        dbc.Col([dbc.Label("Tau",style=label_font)],width={'size':2})]),
    dbc.Row([
        dbc.Col([
                dbc.Input(id='N', value=10,
                          type='number',style=input_style,disabled=False
                          )
                        ],width={'size':2}
                    ),
        dbc.Col([
                dbc.Input(id='beta', value=0.5, type='number',style=input_style
                          ),
                        ],width=2
                    ),
        dbc.Col([
                dbc.Input(id='W', value=3, type='number',style=input_style
                          )
                        ],width=2
                    ),
        dbc.Col([
                dbc.Input(id='S', value=1, type='number',style=input_style
                         )
                        ],width=2
                    ),
        dbc.Col([
                dbc.Input(id='tol', value=10e-3, type='number',style=input_style
                          ),
                        ],width=2
                    ),
        dbc.Col([
                dbc.Input(id='tau', value=0.1, type='number', style=input_style
                       )
                    ], width=2
                    )]),
    dbc.Row([
        dbc.Col([dbc.Label("Integrator", style=label_font)], width={'size': 2, 'offset': 4}),
        dbc.Col([dbc.Label("Final time", style=label_font)], width={'size': 2, 'offset': 4})]),
    dbc.Row(
        [dbc.Col([
            dcc.Dropdown(id='Integrator',options=[{'label': 'ABC2', 'value': 'ABC2'},{'label': 'ABC6', 'value': 'ABC6'}],value='ABC6')], width={'size': 2, 'offset': 4}),
        dbc.Col([
                dbc.Input(id='Final time', value=100, type='number', style=input_style)
                 ],width = {"size":2,'offset':4}
                )]
            ),
        html.Br(),
        dbc.Container(id='nodes',children = [dcc.RangeSlider(id='init',marks={i: 'Node {}'.format(i) for i in range(1, 10)},min=1,max=10,value=[4, 7])]),
        html.Br(),
        dcc.Upload(id='upload-data',children=([html.Button('Upload initial conditions',style=input_style)]),multiple=True),
        html.Button("Clear file",id='clear-data',style=input_style),
        html.Br(),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='momenta', config={'scrollZoom': True, 'showTips': True}),
            dcc.Dropdown(id='site',
                                      options=[],
                                      value=''
                                  ),
                ],width = {"size":5,'offset':1}),
        dbc.Col([
            dcc.Graph(id='energy', config={'scrollZoom': True, 'showTips': True})
               ],width = {"size":5})
            ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='second_moment', config={'scrollZoom': True, 'showTips': True})
                ])
           ]),
    dcc.Store(id='output-data-upload',storage_type='session'),
    html.Div(id='info'),
],fluid=True)

@app.callback([Output('nodes','children'),Output('site','value')], [Input('N','value')])

def giveSlider(N):
    if str(N).isdigit() == False:
        return ['Nop','']
    elif N > 4:
        return [dcc.RangeSlider(id='init',marks={i: 'Node {}'.format(i) for i in range(1, N+1)},min=1,max=N,value=[4, 7]),int(N/2)+1]
    else:
        return ['Nop','']

@app.callback([Output('momenta','figure'),Output('site','options'),Output('energy','figure'),Output('N','disabled'),
               Output('init','disabled'),Output('W','disabled'),Output('second_moment','figure')],
              [Input('beta','value'),Input('W','value'),Input('S','value'),Input('tol','value'),
               Input('tau','value'),Input('Final time','value'),Input('init','value'),Input('site','value'),
               Input('Integrator','value'),Input('output-data-upload', 'data')],
              [State('init','max')])
def makePlot(beta,W,S,tol,tau,t_max,init,site_val,integrator,epsilons,max):
    global df
    if site_val == '':
        raise PreventUpdate
    N = max
    inputs = [N,beta,W,S,tol]
    # print(inputs)
    # print(len([i for i in inputs if str(i).lstrip('-').replace('.','',1).isdigit()]))
    # print(len(inputs))
    if len([i for i in inputs if str(i).lstrip('-').replace('.','',1).isdigit()])==len(inputs) and N > 5:
        N = int(N)
        # # Random list of alpha's
        # epsilon = np.random.uniform(-W / 2, W / 2, size=N)
        # # fixed epsilon
        # epsilon = np.array([0.766602937640871396,
        #                     0.335834373107090300,
        #                     -0.047120909895906671,
        #                     -0.378671947858365998,
        #                     0.213778237119955028,
        #                     -1.283967764250099064,
        #                     -0.786271892038580278,
        #                     -0.734807674784912268,
        #                     -0.465222237470784616,
        #                     -1.345431119907277484, ])
        '''
        INITIAL CONDITIONS
        '''

        # Initial positions / momenta
        q_0 = np.array([0.0] * (N))
        p_0 = np.array([0.0] * (N))

        # Tuning parameter
        # size of wave packet
        L = init[-1] - init[0]

        # start of packet
        start = init[0]

        # tolerance
        eps = tol

        # Function of interval size, start, locations and size of wave packet
        def makeWavepacket(p_0, N, start, size):
            p = np.sqrt(2 * S / L)
            for i in range(start, start + size):
                p_0[i] = p * (-1) ** np.random.randint(0, 2)
            # hard code
            # p_0 = np.array([0., 0., 0., 0., 0.707106781186547573,
            #                     0.707106781186547573, -0.707106781186547573, -0.707106781186547573, 0., 0.])
            # print('Generated initial wave packet: \n {}'.format(pd.DataFrame(p_0)))
            return p_0

        if epsilons is None:
#             print('taking random epsilons')
#             print('\n')
            epsilon = np.random.uniform(-W / 2, W / 2, size=N)
            p_0 = makeWavepacket(p_0, N, start, L)
            outputs = [False,False,False]
        else:
            df = pd.read_json(epsilons)
#             print('taking epsilons from file')
            epsilon = np.array(list(df['Epsilons']))
            p_0 = np.array(list(df['Momenta']))
            q_0 = np.array(list(df['Position']))
            outputs = [True,True,True]

        H_0 = (epsilon / 2) * (q_0 ** 2 + p_0 ** 2) + (beta / 8) * (q_0 ** 2 + p_0 ** 2) ** 2 - np.roll(p_0,
                                                                                                        -1) * p_0 - np.roll(
            q_0, -1) * q_0
        H_0[-1] = (epsilon[-1] / 2) * (q_0[-1] ** 2 + p_0[-1] ** 2) + (beta / 8) * (q_0[-1] ** 2 + p_0[-1] ** 2) ** 2
#         print('Initial Hamiltonian: {}'.format(sum(H_0)))

        S_0 = 0.5 * (q_0 * 2 + p_0 ** 2)

        # print('Hamiltonian {}'.format(sum(H_0)))
        # print('S = {}'.format(sum(S_0)))
        # plt.plot([i for i in range(len(H_0))],H_0)
        # plt.title('Energies for sites')
        # plt.ylabel('Energy')
        # plt.xlabel('Position')

        # Initial evolution vector
        u_0 = list(zip(q_0, p_0))
#         print('U0 = {}'.format(u_0))

        # initial time
        t_0 = 0

        # time vector
        time = np.array([])
        p_ymid = np.array([])
        H = np.array([])
        S = np.array([])
        second_moment = np.array([])
        def LA_t(c, vec):
            # Declare Q's and P's
            qvec = np.array([x[0] for x in vec])
            pvec = np.array([x[1] for x in vec])
            alpha = epsilon + beta * (np.array(qvec) ** 2 + np.array(pvec) ** 2) / 2
            # Iteratives
            qs = qvec * np.cos(c * tau * alpha) + pvec * np.sin(c * tau * alpha)
            ps = pvec * np.cos(c * tau * alpha) - qvec * np.sin(c * tau * alpha)
            return list(zip(qs, ps))

        def LB_t(c, vec):
            qvec = np.array([x[0] for x in vec])
            pvec = np.array([x[1] for x in vec])
            qs = qvec - c * tau * (np.roll(pvec, 1) + np.roll(pvec, -1))
            ps = pvec
            # handle bc's seperately
            qs[0] = qvec[0] - c * tau * pvec[1]
            qs[-1] = qvec[-1] - c * tau * pvec[-2]
            return list(zip(qs, ps))

        def LC_t(d, vec):
            qvec = np.array([x[0] for x in vec])
            pvec = np.array([x[1] for x in vec])
            qs = qvec
            ps = pvec + d * tau * (np.roll(qvec, 1) + np.roll(qvec, -1))
            # handle bc's seperately
            ps[0] = pvec[0] + d * tau * qvec[1]
            ps[-1] = pvec[-1] + d * tau * qvec[-2]
            return list(zip(qs, ps))

        vec_0 = u_0
        w1 = -0.117767998417887e1
        w2 = 0.235573213359357
        w3 = 0.784513610477560
        w0 = 1 - 2 * (w1 + w2 + w3)
        c = 1 / 2
        d = 1

        while t_0 < t_max:

            '''
            Steps here
            '''

            time = np.append(time, t_0)

            # node that u are interested in
            node = int(site_val)-1
            # add its momentum
            p_ymid = np.append(p_ymid, vec_0[node][1])
            if integrator == 'ABC6':
                vec = LA_t(w3 * c, vec_0)
                vec = LB_t(w3 * c, vec)
                vec = LC_t(w3 * d, vec)
                vec = LB_t(w3 * c, vec)
                vec = LA_t(w3 * c, vec)

                vec = LA_t(w2 * c, vec)
                vec = LB_t(w2 * c, vec)
                vec = LC_t(w2 * d, vec)
                vec = LB_t(w2 * c, vec)
                vec = LA_t(w2 * c, vec)

                vec = LA_t(w1 * c, vec)
                vec = LB_t(w1 * c, vec)
                vec = LC_t(w1 * d, vec)
                vec = LB_t(w1 * c, vec)
                vec = LA_t(w1 * c, vec)

                vec = LA_t(w0 * c, vec)
                vec = LB_t(w0 * c, vec)
                vec = LC_t(w0 * d, vec)
                vec = LB_t(w0 * c, vec)
                vec = LA_t(w0 * c, vec)

                vec = LA_t(w1 * c, vec)
                vec = LB_t(w1 * c, vec)
                vec = LC_t(w1 * d, vec)
                vec = LB_t(w1 * c, vec)
                vec = LA_t(w1 * c, vec)

                vec = LA_t(w2 * c, vec)
                vec = LB_t(w2 * c, vec)
                vec = LC_t(w2 * d, vec)
                vec = LB_t(w2 * c, vec)
                vec = LA_t(w2 * c, vec)

                vec = LA_t(w3 * c, vec)
                vec = LB_t(w3 * c, vec)
                vec = LC_t(w3 * d, vec)
                vec = LB_t(w3 * c, vec)
                vec = LA_t(w3 * c, vec)
            else:
                vec = LA_t(c, vec_0)
                vec = LB_t(c, vec)
                vec = LC_t(d, vec)
                vec = LB_t(c, vec)
                vec = LA_t(c, vec)

            vec_0 = vec

            q_0 = np.array([i[0] for i in vec])
            p_0 = np.array([i[1] for i in vec])

            H_new = (epsilon / 2) * (q_0 ** 2 + p_0 ** 2) + (beta / 8) * (q_0 ** 2 + p_0 ** 2) ** 2 - np.roll(p_0,
                                                                                                              -1) * p_0 - np.roll(
                q_0, -1) * q_0
            H_new[-1] = (epsilon[-1] / 2) * (q_0[-1] ** 2 + p_0[-1] ** 2) + (beta / 8) * (q_0[-1] ** 2 + p_0[-1] ** 2) ** 2

            S_new = 0.5 * (q_0 ** 2 + p_0 ** 2)

            S = np.append(S, np.log10(abs((sum(S_new) - sum(S_0)) / sum(S_0))))
            H = np.append(H, np.log10(abs((sum(H_new) - sum(H_0)) / sum(H_0))))

            # new calcs
            norms = S_new / sum(S_new)
            lbar = sum([(index + 1) * i for index, i in enumerate(norms)])
            Part_num = 1 / (sum(norms ** 2))
            #     print(lbar,'center of distribution')
            moment = sum([i * ((index + 1) - lbar) ** 2 for index, i in enumerate(norms)])
            second_moment = np.append(second_moment,moment)

            if abs((sum(S_new) - sum(S_0)) / sum(S_new)) >= eps:
#                 print(abs((sum(S_new) - sum(S_0)) / sum(S_new)))
#                 print(sum(S_new))
#                 print('S not conserved up to {}, broken at time step {}'.format(eps, t_0))
                break
            if abs((sum(H_new) - sum(H_0)) / sum(H_new)) >= eps:
#                 print(abs((sum(H_new) - sum(H_0)) / sum(H_new)))
#                 print(sum(H_new), 'new hamiltonian')
#                 print('H not conserved up to {}, broken at time step {}'.format(eps, t_0))
                break
            t_0 += tau

        df = pd.DataFrame()
        df['Time'] = time
        df['Py'] = p_ymid
        df['second_moment'] = np.log10(second_moment)
        df['relative_energy_error'] = H
        fig = subplots.make_subplots()
        fig['layout'].update(height=300, title='Momenta of site ', title_x=0.5,
                             xaxis_title="Time",
                             yaxis_title="Momentum")
        fig['layout']['margin'] = {'l': 20, 'b': 30, 'r': 10, 't': 50}
        fig.append_trace({'x': df['Time'], 'y': df['Py'], 'type': 'scatter', 'name': 'Susceptible'}, 1, 1)
        fig2 = subplots.make_subplots()
        fig2['layout'].update(height=300, title='Hamiltonian error', title_x=0.5,
                             xaxis_title="Time",
                             yaxis_title="Energy error")
        fig2['layout']['margin'] = {'l': 20, 'b': 30, 'r': 10, 't': 50}
        fig2.append_trace({'x': df['Time'].apply(lambda x: np.log10(x)), 'y': df['relative_energy_error'],
                           'type': 'scatter', 'name': 'Susceptible'}, 1, 1)
        fig3 = subplots.make_subplots()
        fig3['layout'].update(height=500, title='Second moment', title_x=0.5,
                             xaxis_title="Time",
                             yaxis_title="Second moment")
        fig3.append_trace({'x': df['Time'].apply(lambda x: np.log10(x)), 'y': df['second_moment'],
                           'type': 'scatter', 'name': 'Susceptible'}, 1, 1)
        return [fig,[{'label':i,'value':i} for i in range(1,N+1)],fig2,outputs[0],outputs[1],outputs[2],fig3]
    else:
        return [{
                'data': [], 'layout': {
                'height': 500,
                'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10}},},
                [{'label':i,'value':i} for i in range(1,N+1)],
                {'data': [], 'layout': {
                'height': 500,
                'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10}}, },False,False,False,
                {
                'data': [], 'layout': {
                'height': 500,
                'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10}}, }
               ]


def parse_contents(contents, filename, date):
    global df

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
#         print(df)
    except Exception as e:
#         print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df.to_json(),filename,date


@app.callback([Output('output-data-upload', 'data'),Output('info','children')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents,list_of_names, list_of_dates):
#     print(list_of_contents,'listofcontents')
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return [children[0][0],['Loaded {} at'.format(children[0][1]),
                                                         html.H6(datetime.datetime.fromtimestamp(children[0][2]))]]
    else:
        raise PreventUpdate

@app.callback([Output('output-data-upload', 'clear_data')],[Input('clear-data', 'n_clicks')])
def clear_data(n_clicks):
    if n_clicks is not None and n_clicks > 0:
        return [True]
    else:
        return [False]

if __name__ == '__main__':
    app.run_server(debug=True, port=8030)
