import { useState } from 'react';
import { fetchDataset } from '../api/api';

function DataSelector({ onDatasetSelected }) {
    const [dataset, setDataset] = useState("");

    const handleFetch = async () => {
        const response = await fetchDataset(dataset);
        console.log(response.data);
        onDatasetSelected(response.data);
    };

    return (
        <div>
            <h2>Select Dataset</h2>
            <select onChange={(e) => setDataset(e.target.value)}>
                <option value="iris">Iris</option>
                <option value="penguins">Penguins</option>
            </select>
            <button onClick={handleFetch}>Load Dataset</button>
        </div>
    );
}

export default DataSelector;
