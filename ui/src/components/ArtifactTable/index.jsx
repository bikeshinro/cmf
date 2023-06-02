// ArtifactTable.jsx



const ArtifactTable = ({ artifacts }) => {

return (
    <div className="flex flex-col object-cover h-80 w-240 h-screen">
      <div className="overflow-scroll">
        <div className="p-1.5 w-full inline-block align-middle">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-100">
              <tr className="text-xs font-bold text-left text-gray-500 uppercase">
                <th scope="col" className="commit px-6 py-3">Commit</th>
                <th scope="col" className="create_time_since_epoch px-6 py-3">create_time_since_epoch</th>
                <th scope="col" className="Event px-6 py-3">Event</th>
                <th scope="col" className="git_repo px-6 py-3">git_repo</th>
                <th scope="col" className="id px-6 py-3">id</th>
                <th scope="col" className="last_update_time_since_epoch px-6 py-3">last_update_time_since_epoch</th>
                <th scope="col" className="model_framework px-6 py-3">model_framework</th>
                <th scope="col" className="model_name px-6 py-3">model_name</th>
                <th scope="col" className="model_type px-6 py-3">model_type</th>
                <th scope="col" className="name px-6 py-3">name</th>
                <th scope="col" className="original_create_time_since_epoch px-6 py-3">original_create_time_since_epoch</th>
                <th scope="col" className="type px-6 py-3">type</th>
                <th scope="col" className="uri px-6 py-3">uri</th>
                <th scope="col" className="url px-6 py-3">url</th>
              </tr>
            </thead>
            <tbody className="body divide-y divide-gray-200">
              {artifacts.map((data, index) => (
                <tr key={index} className="text-sm font-medium text-gray-800">
                  <td className="px-6 py-4">{data.Commit}</td>
                  <td className="px-6 py-4">{data.create_time_since_epoch}</td>
                  <td className="px-6 py-4">{data.event}</td>
                  <td className="px-6 py-4">{data.git_repo}</td>
                  <td className="px-6 py-4">{data.id}</td>
                  <td className="px-6 py-4">{data.last_update_time_since_epoch}</td>
                  <td className="px-6 py-4">{data.model_framework}</td>
                  <td className="px-6 py-4">{data.model_name}</td>
                  <td className="px-6 py-4">{data.model_type}</td>
                  <td className="px-6 py-4">{data.name}</td>
                  <td className="px-6 py-4">{data.original_create_time_since_epoch}</td>
                  <td className="px-6 py-4">{data.type}</td>
                  <td className="px-6 py-4">{data.uri}</td>
                  <td className="px-6 py-4">{data.url}</td>  
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ArtifactTable;
