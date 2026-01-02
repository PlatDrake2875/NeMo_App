import PropTypes from "prop-types";
import { Button } from "../ui/button";
import { Separator } from "../ui/separator";
import {
  ArrowLeft,
  Database,
  Download,
  FolderOpen,
  GitBranch,
  LayoutDashboard,
  Moon,
  Sun,
} from "lucide-react";
import { cn } from "../../lib/utils";

const navItems = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "raw", label: "Raw Datasets", icon: FolderOpen },
  { id: "pipeline", label: "Preprocessing", icon: GitBranch },
  { id: "processed", label: "Processed Datasets", icon: Database },
  { id: "huggingface", label: "HuggingFace Import", icon: Download },
];

export function RAGHubSidebar({
  currentView,
  onViewChange,
  onBack,
  isDarkMode,
  toggleTheme,
}) {
  return (
    <aside className="hidden md:flex w-56 border-r bg-muted/30 flex-col h-full">
      {/* Back Button */}
      <div className="p-4">
        <Button
          variant="ghost"
          onClick={onBack}
          className="w-full justify-start gap-2"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Chat
        </Button>
      </div>

      <Separator />

      {/* Navigation */}
      <nav className="flex-1 p-2 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentView === item.id;

          return (
            <Button
              key={item.id}
              variant={isActive ? "secondary" : "ghost"}
              onClick={() => onViewChange(item.id)}
              className={cn(
                "w-full justify-start gap-2",
                isActive && "bg-accent"
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Button>
          );
        })}
      </nav>

      <Separator />

      {/* Theme Toggle */}
      <div className="p-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleTheme}
          className="w-full justify-start gap-2"
        >
          {isDarkMode ? (
            <>
              <Sun className="h-4 w-4" />
              Light Mode
            </>
          ) : (
            <>
              <Moon className="h-4 w-4" />
              Dark Mode
            </>
          )}
        </Button>
      </div>
    </aside>
  );
}

RAGHubSidebar.propTypes = {
  currentView: PropTypes.string.isRequired,
  onViewChange: PropTypes.func.isRequired,
  onBack: PropTypes.func.isRequired,
  isDarkMode: PropTypes.bool,
  toggleTheme: PropTypes.func,
};

export default RAGHubSidebar;
