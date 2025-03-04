import axios from 'axios';

const API_URL = "http://localhost:5000";

export const fetchDataset = (datasetName) => {
    return axios.post(`${API_URL}/datasets/get`, { datasetName });
};

export const predictLogisticRegression = (features) => {
    return axios.post(`${API_URL}/models/logistic-regression`, { features });
};

export const predictLinearRegression = (features) => {
    return axios.post(`${API_URL}/models/linear-regression`, { features });
};
