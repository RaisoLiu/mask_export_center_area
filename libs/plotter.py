import plotly.graph_objects as go
import plotly.subplots as sp
class Plotter:
    @staticmethod
    def plot_area(area):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(area))),
                        y=area, mode='lines', name=f'area'))

        return fig

    @staticmethod
    def plot_coordinates(x_list, y_list):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(x_list))),
                        y=x_list, mode='lines', name=f'x'))
        fig.add_trace(go.Scatter(x=list(range(len(y_list))),
                        y=y_list, mode='lines', name=f'y'))
      
        fig.update_layout(title_text="XY-coordinates of the colors over time")
        return fig


