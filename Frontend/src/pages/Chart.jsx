import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const Chart = ({ data }) => {
  const chartData = {
    labels: data.labels,
    datasets: [
      {
        label: 'Actual Data',
        data: data.values,
        borderColor: 'blue',
        backgroundColor: 'blue',
        pointRadius: 5,
        fill: false,
      },
      {
        label: 'Regression Line',
        data: data.predicted,
        borderColor: 'red',
        backgroundColor: 'red',
        pointRadius: 0,
        borderWidth: 2,
        fill: false,
      },
    ],
  };

  return <Line data={chartData} />;
};

export default Chart;
