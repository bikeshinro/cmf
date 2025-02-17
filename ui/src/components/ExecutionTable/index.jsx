//ExecutionTable.jsx
import React, { useState, useEffect } from 'react';
const ExecutionTable = ({ executions }) => {

const [searchQuery, setSearchQuery] = useState('');
const [currentPage, setCurrentPage] = useState(1);
const [itemsPerPage] = useState(5); // Number of items to display per page
const [sortBy, setSortBy] = useState(null); // Property to sort by
const [sortOrder, setSortOrder] = useState('asc'); // Sort order ('asc' or 'desc')
const [expandedRow, setExpandedRow] = useState(null);  
const handleSearchChange = (event) => {
    setSearchQuery(event.target.value);
  };

const handlePageChange = (page) => {
    setCurrentPage(page);
  };


const handleSort = (property) => {
    if (sortBy === property) {
      // If currently sorted by the same property, toggle sort order
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      // If sorting by a new property, set it to ascending order by default
      setSortBy(property);
      setSortOrder('asc');
    }
  };
const consistentColumns = [];

const filteredData = executions.filter((item) =>
    (item.Context_Type && item.Context_Type.toLowerCase().includes(searchQuery.toLowerCase()))
    || (item.Execution && item.Execution.toLowerCase().includes(searchQuery.toLowerCase()))
  );

// eslint-disable-next-line
const sortedData = filteredData.sort((a, b) => {
    const aValue = a[sortBy];
    const bValue = b[sortBy];

    if (aValue < bValue) return sortOrder === 'asc' ? -1 : 1;
    if (aValue > bValue) return sortOrder === 'asc' ? 1 : -1;
    return 0;
  });

const totalPages = Math.ceil(filteredData.length / itemsPerPage);
const indexOfLastItem = currentPage * itemsPerPage;
const indexOfFirstItem = indexOfLastItem - itemsPerPage;
const currentItems = filteredData.slice(indexOfFirstItem, indexOfLastItem);

useEffect(() => {
    setCurrentPage(1); // Reset current page to 1 when search query changes
  }, [searchQuery]);

const toggleRow = (rowId) => {
    if (expandedRow === rowId) {
      setExpandedRow(null);
    } else {
      setExpandedRow(rowId);
    }
  };


return (
    <div className="flex flex-col">
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '1rem' }}>
      <input
        type="text"
        value={searchQuery}
        onChange={handleSearchChange}
        placeholder="Search..."
        style={{ marginRight: '1rem', padding: '0.5rem',border: '1px solid #ccc' }}
      />
      </div>
      <div className="overflow-x-auto">
        <div className="p-1.5 w-full inline-block align-middle">
          <table className="min-w-full divide-y divide-gray-200" id="mytable">
            <thead className="bg-gray-100">
              <tr className="text-xs font-bold text-left text-gray-500 uppercase">
                <th scope="col" className="px-6 py-3"></th>
                <th scope="col" className="px-6 py-3 id">id</th>
                <th scope="col" onClick={() => handleSort('Context_Type')} className="px-6 py-3 Context_Type">Context_Type
              {sortBy === 'Context_Type' && sortOrder === 'asc' && '▲'}
              {sortBy === 'Context_Type' && sortOrder === 'desc' && '▼'}</th>
                <th scope="col" className="px-6 py-3 Execution">Execution</th>
                <th scope="col" className="px-6 py-3 Git_Repo">Git_Repo</th>
                <th scope="col" className="px-6 py-3 Git_Start_Commit">Git_Start_Commit</th>
                <th scope="col" className="px-6 py-3 Pipeline_Type">Pipeline_Type</th>
              </tr>
            </thead>
            <tbody className="body divide-y divide-gray-200">
              {currentItems.map((data, index) => (
                <React.Fragment key={index}>
                <tr key={index} onClick={() => toggleRow(index)} className="text-sm font-medium text-gray-800">
                  <td classname="px-6 py-4">{expandedRow === index ? '-' : '+'}</td>
                  <td className="px-6 py-4">{data.id}</td>
                  <td className="px-6 py-4">{data.Context_Type}</td>
                  <td className="px-6 py-4">{data.Execution}</td>
                  <td className="px-6 py-4">{data.Git_Repo}</td>
                  <td className="px-6 py-4">{data.Git_Start_Commit}</td>
                  <td className="px-6 py-4">{data.Pipeline_Type}</td>
                </tr>
                {expandedRow === index &&  (
                <tr>
                   <td colSpan='4'>
                    <table className="expanded-table">
             <tbody>
            {Object.entries(data).map(([key, value]) => {
               if (!consistentColumns.includes(key) && value != null) {
                  return (  
                    <React.Fragment key={key}>
                      <tr>
                         <td key={key}>{key}</td>
                         <td key={value}>{value  ? value :"Null"}</td>
                      </tr>
                     </React.Fragment>
                         );
                        }
                       return null;
                     })}
               </tbody>
                   </table>
                  </td>
                </tr>
              )}
            </React.Fragment>
              ))}
            </tbody>
          </table>
          </div>
          <div>
        <button
          disabled={currentPage === 1}
          onClick={() => handlePageChange(currentPage - 1)}
        >
          Previous
        </button>
        {Array.from({ length: totalPages }, (_, index) => index + 1).map(
          (page) => (
            <button
              key={page}
              onClick={() => handlePageChange(page)}
              disabled={currentPage === page}
            >
              {page}
            </button>
          )
        )}
        <button
          disabled={currentPage === totalPages}
          onClick={() => handlePageChange(currentPage + 1)}
        >
          Next
        </button>
      </div>
       </div>
      </div>
  );
};


export default ExecutionTable;
