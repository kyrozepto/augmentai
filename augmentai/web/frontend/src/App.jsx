import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Datasets from './pages/Datasets'
import PolicyBuilder from './pages/PolicyBuilder'
import Search from './pages/Search'
import Chat from './pages/Chat'
import Ablation from './pages/Ablation'
import Diff from './pages/Diff'
import Repair from './pages/Repair'
import Curriculum from './pages/Curriculum'
import Shift from './pages/Shift'
import './index.css'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="dataset" element={<Datasets />} />
          <Route path="policy" element={<PolicyBuilder />} />
          <Route path="search" element={<Search />} />
          <Route path="chat" element={<Chat />} />
          <Route path="ablation" element={<Ablation />} />
          <Route path="diff" element={<Diff />} />
          <Route path="repair" element={<Repair />} />
          <Route path="curriculum" element={<Curriculum />} />
          <Route path="shift" element={<Shift />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App



