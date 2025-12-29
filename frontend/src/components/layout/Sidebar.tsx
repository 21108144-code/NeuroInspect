import { NavLink, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import {
    LayoutDashboard,
    ScanSearch,
    BarChart3,
    GitBranch,
    Brain,
    Settings,
    ChevronLeft,
    ChevronRight,
} from 'lucide-react'
import { useState } from 'react'

const navItems = [
    {
        title: 'Dashboard',
        href: '/',
        icon: LayoutDashboard,
        description: 'Overview and KPIs',
    },
    {
        title: 'Inspection',
        href: '/inspection',
        icon: ScanSearch,
        description: 'Analyze images',
    },
    {
        title: 'Analytics',
        href: '/analytics',
        icon: BarChart3,
        description: 'Defect statistics',
    },
    {
        title: 'Root Cause',
        href: '/root-cause',
        icon: GitBranch,
        description: 'Pattern analysis',
    },
    {
        title: 'Explainability',
        href: '/explainability',
        icon: Brain,
        description: 'AI explanations',
    },
    {
        title: 'Settings',
        href: '/settings',
        icon: Settings,
        description: 'Configuration',
    },
]

export function Sidebar() {
    const [collapsed, setCollapsed] = useState(false)
    const location = useLocation()

    return (
        <aside
            className={cn(
                'fixed left-0 top-0 z-40 h-screen bg-card border-r border-border transition-all duration-300',
                collapsed ? 'w-16' : 'w-64'
            )}
        >
            {/* Logo */}
            <div className="flex items-center justify-between h-16 px-4 border-b border-border">
                {!collapsed && (
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                            <Brain className="w-5 h-5 text-white" />
                        </div>
                        <span className="font-bold text-lg text-gradient">NeuroInspect</span>
                    </div>
                )}
                <button
                    onClick={() => setCollapsed(!collapsed)}
                    className="p-1.5 rounded-md hover:bg-muted transition-colors"
                >
                    {collapsed ? (
                        <ChevronRight className="w-4 h-4" />
                    ) : (
                        <ChevronLeft className="w-4 h-4" />
                    )}
                </button>
            </div>

            {/* Navigation */}
            <nav className="p-2 space-y-1">
                {navItems.map((item) => {
                    const isActive = location.pathname === item.href
                    return (
                        <NavLink
                            key={item.href}
                            to={item.href}
                            className={cn(
                                'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200',
                                'hover:bg-muted group',
                                isActive
                                    ? 'bg-primary/10 text-primary border border-primary/20'
                                    : 'text-muted-foreground hover:text-foreground'
                            )}
                        >
                            <item.icon
                                className={cn(
                                    'w-5 h-5 flex-shrink-0 transition-colors',
                                    isActive ? 'text-primary' : 'group-hover:text-foreground'
                                )}
                            />
                            {!collapsed && (
                                <div className="overflow-hidden">
                                    <div className="font-medium text-sm">{item.title}</div>
                                    <div className="text-xs text-muted-foreground truncate">
                                        {item.description}
                                    </div>
                                </div>
                            )}
                        </NavLink>
                    )
                })}
            </nav>

            {/* Footer */}
            {!collapsed && (
                <div className="absolute bottom-4 left-4 right-4">
                    <div className="p-3 rounded-lg bg-muted/50 border border-border">
                        <p className="text-xs text-muted-foreground">
                            Industrial AI Inspection
                        </p>
                        <p className="text-xs font-medium mt-1">v1.0.0</p>
                    </div>
                </div>
            )}
        </aside>
    )
}
