import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { ThemeProvider } from '@/components/layout/ThemeProvider'
import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'
import Dashboard from '@/pages/Dashboard'
import Inspection from '@/pages/Inspection'
import Analytics from '@/pages/Analytics'
import RootCause from '@/pages/RootCause'
import Explainability from '@/pages/Explainability'
import Settings from '@/pages/Settings'

function AppLayout({ children }: { children: React.ReactNode }) {
    return (
        <div className="min-h-screen bg-background">
            <Sidebar />
            <div className="ml-64 transition-all duration-300">
                {children}
            </div>
        </div>
    )
}

export default function App() {
    return (
        <ThemeProvider defaultTheme="dark">
            <BrowserRouter>
                <AppLayout>
                    <Routes>
                        <Route
                            path="/"
                            element={
                                <>
                                    <Header title="Dashboard" description="Real-time inspection overview" />
                                    <Dashboard />
                                </>
                            }
                        />
                        <Route
                            path="/inspection"
                            element={
                                <>
                                    <Header title="Inspection" description="Analyze images for defects" />
                                    <Inspection />
                                </>
                            }
                        />
                        <Route
                            path="/analytics"
                            element={
                                <>
                                    <Header title="Analytics" description="Defect statistics and trends" />
                                    <Analytics />
                                </>
                            }
                        />
                        <Route
                            path="/root-cause"
                            element={
                                <>
                                    <Header title="Root Cause Analysis" description="Pattern clustering and insights" />
                                    <RootCause />
                                </>
                            }
                        />
                        <Route
                            path="/explainability"
                            element={
                                <>
                                    <Header title="Explainability" description="AI decision explanations" />
                                    <Explainability />
                                </>
                            }
                        />
                        <Route
                            path="/settings"
                            element={
                                <>
                                    <Header title="Settings" description="System configuration" />
                                    <Settings />
                                </>
                            }
                        />
                    </Routes>
                </AppLayout>
            </BrowserRouter>
        </ThemeProvider>
    )
}
