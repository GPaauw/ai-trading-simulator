import { useState } from 'react'
import LoginPage from './pages/LoginPage'
import Dashboard from './pages/Dashboard'
import { api } from './api'

export default function App() {
  const [loggedIn, setLoggedIn] = useState(api.isAuthenticated())

  if (!loggedIn) {
    return <LoginPage onLogin={() => setLoggedIn(true)} />
  }

  return (
    <Dashboard
      onLogout={() => {
        api.clearToken()
        setLoggedIn(false)
      }}
    />
  )
}
