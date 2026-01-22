import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Live from "./pages/Live";
import Simulation from "./pages/Simulation";
import Enroll from "./pages/Enroll";
import Results from "./pages/Results";

import "./styles/layout.css";

function Layout({ children }) {
  return (
    <div className="appShell">
      <aside className="sidebar">
        <h2 className="sidebarTitle">FYP</h2>

        <nav className="nav">
          <NavLink
            to="/"
            end
            className={({ isActive }) => (isActive ? "navLink navLinkActive" : "navLink")}
          >
            Live
          </NavLink>

          <NavLink
            to="/enroll"
            className={({ isActive }) => (isActive ? "navLink navLinkActive" : "navLink")}
          >
            Enroll
          </NavLink>

          <NavLink
            to="/simulation"
            className={({ isActive }) => (isActive ? "navLink navLinkActive" : "navLink")}
          >
            Simulation
          </NavLink>

          <NavLink
            to="/results"
            className={({ isActive }) => (isActive ? "navLink navLinkActive" : "navLink")}
          >
            Results
          </NavLink>
        </nav>
      </aside>

      <main className="main">{children}</main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout><Live /></Layout>} />
        <Route path="/enroll" element={<Layout><Enroll /></Layout>} />
        <Route path="/simulation" element={<Layout><Simulation /></Layout>} />
        <Route path="/results" element={<Layout><Results /></Layout>} />
      </Routes>
    </BrowserRouter>
  );
}
